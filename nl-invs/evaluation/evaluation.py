
import logging
import os
from typing import Union
import importlib
import sys, platform

if platform.node() == 'lars-HP-ENVY-Laptop-15-ep0xxx':
    sys.path.append('/home/lars/Outliers/nonlinear-outlier-synthesis/scripts')
else:
    sys.path.append('/storage/homefs/ld20d359/nonlinear-outlier-synthesis/scripts')

import numpy as np
import matplotlib.pyplot as plt

# Torch imports
from torch import nn
import torch
from torch.utils.data import DataLoader

# Ignite imports
from ignite.utils import setup_logger
import ignite.distributed as idist

# Local imports
from nl_invariants.trainer import Trainer, find_number_of_invariants, _build_model, load, build_optimizer, build_engine
from nl_invariants.utils import expanduservars
from nl_invariants.model import ConditionalVolumePreservingNet

from dream_ood import get_class_names

LOGGER = logging.getLogger(__name__)
Model = Union[ConditionalVolumePreservingNet, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


@torch.no_grad()
def get_embeddings(model: Model, train_loader, num_classes, number_of_invariants):
    model.eval()

    labels = train_loader.dataset.tensors[1]
    mse = 0
    mse_per_class = np.zeros(num_classes)

    embeddings = []
    for i, (x, y) in enumerate(train_loader):
        
        device = idist.device()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred_y = model(x, y)

        mse += torch.mean(pred_y[:, -number_of_invariants:] ** 2)

        # for j in range(x.shape[0]):
        #     mse_per_class[y[j].item()] += torch.mean(pred_y[j, -number_of_invariants:] ** 2)

        embeddings.append(pred_y.detach().cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(embeddings.shape)
    print(mse, mse / len(train_loader))

    if np.sum(mse_per_class) > 0:
        plt.figure()
        plt.bar(range(num_classes), mse_per_class)
        plt.savefig("mse_per_class.png")
        plt.close()

        # Get 3 highest mse classes
        highest_mse_classes = np.argsort(mse_per_class)[::-1][:3]
        for i in highest_mse_classes:
            class Object(object):
                pass

            opt = Object()
            opt.id_data = 'c100'
            name = get_class_names(opt)[i]
            print(f"Class {i} ({name}): {mse_per_class[i]}")

    # Group embeddings by class label
    grouped_embeddings = {}

    for i in range(num_classes):
        grouped_embeddings[str(i)] = embeddings[labels == i]

    print(len(grouped_embeddings))

    return grouped_embeddings


def sample_outliers(embeddings, num_classes, network, number_of_invariants, params):
    generated_samples = np.zeros((num_classes, 100, embeddings['0'].shape[-1]))

    for c in range(num_classes):
        class_embs = torch.as_tensor(embeddings[str(c)])
        print(f"Class {c}", flush=True)

        mean = torch.mean(class_embs, dim=0)
        cov = torch.cov(class_embs.T)
        # Add small value to diagonal to make it positive definite
        cov = cov + params["reg_weight"] * torch.eye(cov.shape[0])

        distribution = torch.distributions.MultivariateNormal(mean, cov.to(torch.float32))

        ood_sample_list = []
        for i in range(10):
            samples = distribution.rsample((3330,))
            prob_density = distribution.log_prob(samples)

            cur_samples, index_prob = torch.topk(-prob_density, 10)
            ood_samples = samples[index_prob].to(idist.device())
            clss = torch.full((len(ood_samples),), c, dtype=torch.long).to(idist.device())

            ood_sample_list.append(network.reverse(ood_samples, clss).detach().cpu().numpy())

        ood_sample_list = np.concatenate(ood_sample_list, axis=0)
        generated_samples[c] = ood_sample_list

    print(generated_samples.shape)
    np.save("generated_samples_" + params["dataset_file"][9:] + "_" + str(params["reg_weight"]) + ".npy", generated_samples)


def run_eval(local_rank: int, params: dict):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    os.makedirs(output_path, exist_ok=True)
    
    LOGGER.info("%d GPUs available", torch.cuda.device_count())
    
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_dataset = dataset_module.training_dataset(params["preprocessing"], params["train_fraction"])  # type: ignore

    num_classes = len(np.unique(train_dataset[:][1]))
    LOGGER.info("Number of classes: %d", num_classes)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

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

    # Sample outliers
    embeddings = get_embeddings(trainer.model, train_loader, num_classes, number_of_invariants)
    sample_outliers(embeddings, num_classes, trainer.model, number_of_invariants, params)
    