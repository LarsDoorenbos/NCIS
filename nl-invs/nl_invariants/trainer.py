
from dataclasses import dataclass
import logging
import importlib
import os
from typing import Optional, Dict, Union, Any, cast

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# Torch imports
from torch import nn
from torch import Tensor
import torch
from torch.utils.data import DataLoader, Subset

# Ignite imports
from ignite.engine import Engine, Events
import ignite.distributed as idist
from ignite.utils import setup_logger
from ignite.metrics import Frequency, MeanSquaredError
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import WandBLogger
from ignite.contrib.metrics import GpuInfo

# Local imports
from .utils import archive_code, expanduservars
from .model import build_model, ConditionalVolumePreservingNet
from .optimizer import build_optimizer

LOGGER = logging.getLogger(__name__)
Model = Union[ConditionalVolumePreservingNet, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def _flatten(m: Model) -> ConditionalVolumePreservingNet:
    if isinstance(m, ConditionalVolumePreservingNet):
        return m
    elif isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        return cast(ConditionalVolumePreservingNet, m.module)
    else:
        raise TypeError("type(m) should be one of (ConditionalVolumePreservingNet, DataParallel, DistributedDataParallel)")
    

def _loader_subset(loader: DataLoader, num_images: int) -> DataLoader:
    dataset = loader.dataset
    lng = len(dataset)
    return DataLoader(
        Subset(dataset, range(0, lng - lng % num_images, lng // num_images)),
        batch_size=loader.batch_size,
        shuffle=False
    )


def find_number_of_invariants(dataset, pca_variance_percentage):
    data, labels = dataset[:]
    
    cl_invs = []
    for cl in range(len(np.unique(labels))):
        data_cl = data[labels == cl]

        pca = PCA()
        pca_variance_ratio = 1 - (pca_variance_percentage / 100)
        _ = pca.fit_transform(data_cl)

        number_of_invariants = np.where(np.cumsum(pca.explained_variance_ratio_) > pca_variance_ratio)[0][0]
        number_of_invariants = data.shape[1] - number_of_invariants
        
        cl_invs.append(number_of_invariants)

    number_of_invariants = np.mean(np.array(cl_invs)).astype(int)
    LOGGER.info("Number of invariants: %d", number_of_invariants)

    return number_of_invariants


@dataclass
class Trainer:
    
    model: ConditionalVolumePreservingNet
    optimizer: torch.optim.Optimizer
    lr_scheduler: Union[torch.optim.lr_scheduler.LambdaLR, None]
    invariants: int

    @property
    def flat_model(self):
        """View of the model without DataParallel wrappers."""
        return _flatten(self.model)

    def train_step(self, engine: Engine, batch) -> dict:

        x, y = batch

        self.model.train()
        
        device = idist.device()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = x.shape[0]

        pred_y = self.model(x, y)

        inv_loss = torch.mean(pred_y[:, -self.invariants:] ** 2)

        self.optimizer.zero_grad()
        inv_loss.backward()
        self.optimizer.step()

        inv_loss = inv_loss.item()

        pred_y = pred_y.detach().clone()
        pred_y[:, -self.invariants:] = 0.0

        pred_x = self.flat_model.reverse(pred_y, y)
        rec_loss = torch.nn.functional.mse_loss(pred_x, x)

        self.optimizer.zero_grad()
        rec_loss.backward()
        self.optimizer.step()

        rec_loss = rec_loss.item()
        
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()
        else:
            lr = self.optimizer.defaults['lr']

        return {"num_items": batch_size, "lr": lr, "inv_loss": inv_loss, "rec_loss": rec_loss}

    @torch.no_grad()
    def test_step(self, _: Engine, batch: Tensor) -> Dict[str, Any]:

        x, y = batch
        x = x.to(idist.device())
        y = y.to(idist.device())

        self.model.eval()
        pred_y = self.model(x, y)

        pred_y = pred_y[:, -self.invariants:]
        target = torch.zeros_like(pred_y)

        return {'y': target, 'y_pred': pred_y}

    def objects_to_save(self, engine: Optional[Engine] = None) -> Dict[str, Any]:
        to_save: Dict[str, Any] = {
            "model": self.flat_model,
            "optimizer": self.optimizer,
            "scheduler": self.lr_scheduler
        }

        if engine is not None:
            to_save["engine"] = engine

        return to_save


def build_engine(trainer: Trainer, output_path: str, train_loader: DataLoader, params: dict) -> Engine:
    engine = Engine(trainer.train_step)
    frequency_metric = Frequency(output_transform=lambda x: x["num_items"])
    frequency_metric.attach(engine, "imgs/s", Events.ITERATION_COMPLETED)
    GpuInfo().attach(engine, "gpu")

    engine_test = Engine(trainer.test_step)
    MeanSquaredError().attach(engine_test, "mse")

    if idist.get_local_rank() == 0:
        if params["use_logger"]:
            wandb_logger = WandBLogger(project='nl-dream', name = params["dataset_file"][9:] + '_' + params["preprocessing"], config=params)

            wandb_logger.attach_output_handler(
                engine,
                event_name=Events.ITERATION_COMPLETED(every=100),
                tag="training",
                output_transform=lambda x: x,
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

            wandb_logger.attach_output_handler(
                engine_test,
                Events.EPOCH_COMPLETED,
                tag="testing",
                metric_names=["mse"],
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

        else: 
            wandb_logger = None

        checkpoint_handler = ModelCheckpoint(
            output_path,
            "model",
            n_saved=1,
            require_empty=False,
            score_function=None,
            score_name=None
        )

        checkpoint_best = ModelCheckpoint(
            output_path,
            "best",
            n_saved=1,
            require_empty=False,
            score_function=lambda engine: -engine.state.metrics["mse"],
            score_name='negmse',
            global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        )

    # Display some info every 100 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=100))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_info(engine: Engine):
        LOGGER.info(
            "epoch=%d, iter=%d, speed=%.2fimg/s, rec_loss=%.4g, inv_loss=%.4g, gpu:0 util=%.2f%%",
            engine.state.epoch,
            engine.state.iteration,
            engine.state.metrics["imgs/s"],
            engine.state.output["rec_loss"],
            engine.state.output["inv_loss"],
            engine.state.metrics["gpu:0 util(%)"]
        )

    # Save model every 1000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=5000))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_model(engine: Engine):
        checkpoint_handler(engine, trainer.objects_to_save(engine))

    # Compute the mse/auc score
    @engine.on(Events.EPOCH_COMPLETED(every=params["val_freq"]))
    def test(_: Engine):
        LOGGER.info("MSE&AUC computation...")

        val_loader = _loader_subset(train_loader, 1000)
        engine_test.run(val_loader)

        checkpoint_best(engine_test, trainer.objects_to_save(engine))
                    
        LOGGER.info("MSE: %f", engine_test.state.metrics["mse"])

    return engine


def load(filename: str, trainer: Trainer, engine: Engine):
    LOGGER.info("Loading state from %s...", filename)
    state = torch.load(filename, map_location=idist.device())
    to_load = trainer.objects_to_save(engine)
    ModelCheckpoint.load_objects(to_load, state)


def _build_model(input_dimensionality: int, num_classes: int, params: dict) -> Model:
    model: Model = build_model(
        dim = input_dimensionality,
        num_layers = params["num_layers"],
        num_classes = num_classes,
        channel_mults = params["channel_mults"]
    ).to(idist.device())

    # Wrap the model in DataParallel for parallel processing
    if params["multigpu"]:
        model = nn.DataParallel(model)

    return model


def _get_datasets(params: dict):
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_dataset = dataset_module.training_dataset(params["preprocessing"], params["train_fraction"])  # type: ignore

    num_classes = len(np.unique(train_dataset[:][1]))
    LOGGER.info("Number of classes: %d", num_classes)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    
    return train_loader, num_classes


def run_train(params: dict):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    os.makedirs(output_path, exist_ok=True)
    archive_code(output_path)

    LOGGER.info("%d GPUs available", torch.cuda.device_count())
    LOGGER.info("bs %d, lr %f, pca_var %f", params["batch_size"], params["optim"]["learning_rate"], params["pca_variance_percentage"])

    train_loader, num_classes = _get_datasets(params)

    # Build the model, optimizer, trainer and training engine
    input_dimensionality = train_loader.dataset[0][0].shape[0]
    LOGGER.info("%d dimensions", input_dimensionality)
    
    model = _build_model(input_dimensionality, num_classes, params)

    optimizer_staff = build_optimizer(params, model, train_loader)
    optimizer = optimizer_staff['optimizer']
    lr_scheduler = optimizer_staff['lr_scheduler']

    number_of_invariants = find_number_of_invariants(train_loader.dataset, params["pca_variance_percentage"])

    trainer = Trainer(model, optimizer, lr_scheduler, number_of_invariants)
    engine = build_engine(trainer, output_path, train_loader, params=params)
    
    # Load a model (if requested in params.yml) to continue training from it
    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        load(load_from, trainer=trainer, engine=engine)

    engine.run(train_loader, max_epochs=params["max_epochs"])