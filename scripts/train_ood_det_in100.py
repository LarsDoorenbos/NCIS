# -*- coding: utf-8 -*-
import numpy as np
import os, sys, platform

import argparse
import time
import torch

import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

if platform.node() == 'lars-HP-ENVY-Laptop-15-ep0xxx':
    sys.path.append('/home/lars/Outliers/nonlinear-outlier-synthesis/')
else:
    sys.path.append('/storage/homefs/ld20d359/nonlinear-outlier-synthesis')

from utils.validation_dataset import validation_split

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='in100',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='rn34')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=80, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=80, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
parser.add_argument('--suffix', type=str, default='')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--m_in', type=float, default=-25.,
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--score', type=str, default='energy', help='OE|energy')
parser.add_argument('--add_slope', type=int, default=0)
parser.add_argument('--add_class', type=int, default=0)
parser.add_argument('--vanilla', type=int, default=1)
parser.add_argument('--oe', type=int, default=0)
parser.add_argument('--T', type=float, default=1.0)
parser.add_argument('--my_info', type=str, default='')
parser.add_argument('--cutmix', type=int, default=0)
parser.add_argument('--augmix', type=int, default=0)
parser.add_argument('--vos', type=int, default=0)
parser.add_argument('--gan', type=int, default=0)
parser.add_argument('--r50', type=int, default=0)
parser.add_argument('--godin', type=int, default=0)
parser.add_argument('--deepaugment', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--additional_info', type=str, default='')
parser.add_argument('--energy_weight', type=float, default=1)  # change this to 19.2 if you are using cifar-100.
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')


# energy reg
parser.add_argument('--start_epoch', type=int, default=40)
parser.add_argument('--sample_number', type=int, default=1000)
parser.add_argument('--select', type=int, default=1)
parser.add_argument('--sample_from', type=int, default=10000)
parser.add_argument('--loss_weight', type=float, default=0.1)
args = parser.parse_args()


from resnet import ResNet_Model

if args.score == 'OE':
    save_info = 'oe_tune'
elif args.score == 'energy':
    save_info = 'energy_ft_sd'


args.save = args.save + save_info
if os.path.isdir(args.save) == False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(args.seed)

traindir = os.path.join(os.path.expandvars("${TMPDIR}"), "imgnet_100_train")
valdir = os.path.join(os.path.expandvars("${TMPDIR}"), "imgnet_100_val")


if args.model == 'clip':
    normalize = trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711))
else:
    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

if args.augmix:
    train_data_in = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.AugMix(),
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ])
        )
else:
    train_data_in = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))
if args.deepaugment:
    edsr_dataset = torchvision.datasets.ImageFolder(
        '/nobackup-fast/dataset/my_xfdu/deepaugment/imagenet-r/DeepAugment/EDSR/',
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))

    cae_dataset = torchvision.datasets.ImageFolder(
        '/nobackup-fast/dataset/my_xfdu/deepaugment/imagenet-r/DeepAugment/CAE/',
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))
    train_data_in = torch.utils.data.ConcatDataset([train_data_in, edsr_dataset, cae_dataset])
test_data = torchvision.datasets.ImageFolder(
    valdir,
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))
num_classes = 100


calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'



ood_data = dset.ImageFolder(root=args.my_info,
                            transform=trn.Compose([trn.RandomResizedCrop(224),
trn.RandomHorizontalFlip(),
trn.ToTensor(),
normalize,]))

print(len(train_data_in), len(ood_data), len(test_data))

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model == 'r50':
    net = ResNet_Model(name='resnet50', num_classes=num_classes)
elif args.model == 'clip':
    net = CLIP_ft(num_classes=num_classes, layers=10)
elif args.model == 'convnext':
    net = ResNet_Model(name='convnext_base', num_classes=num_classes)
elif args.model == 'vit':
    net = ResNet_Model(name='vit_b16', num_classes=num_classes)
else:
    net = ResNet_Model(name='resnet34', num_classes=num_classes)

num_of_parameters = sum(map(torch.numel, net.parameters()))
print(f"Trainable params: {num_of_parameters}")


# Restore model
model_found = False
if args.load != '':
    pretrained_weights = torch.load(args.load)
    # breakpoint()
    for item in list(pretrained_weights.keys()):
        pretrained_weights[item[7:]] = pretrained_weights[item]
        del pretrained_weights[item]
    net.load_state_dict(pretrained_weights, strict=True)

logistic_regression = torch.nn.Sequential(
    torch.nn.Linear(1, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 2)
).cuda()

if args.ngpu >= 1:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    logistic_regression = torch.nn.DataParallel(logistic_regression, device_ids=list(range(args.ngpu)))

if args.model == 'clip':
    # Only fine-tune the last two blocks of the transformer
    for i in range(len(net.model.transformer.resblocks)):
        if i < 10:
            for param in net.model.transformer.resblocks[i].parameters():
                param.requires_grad = False

    param_list = [
        {"params": net.ln_post.parameters()},
        {"params": net.model.transformer.parameters()},
        {"params": net.proj},
        {"params": logistic_regression.parameters()}]
    optimizer = torch.optim.SGD(param_list,
        state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)
else:
    optimizer = torch.optim.SGD(
        list(net.parameters()) + list(logistic_regression.parameters()),
        state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

if args.ngpu > 0:
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

# /////////////// Training ///////////////
criterion = torch.nn.CrossEntropyLoss()







def train_permute():
    net.train()  # enter train mode
    loss_avg = 0.0
    loss_energy_avg = 0.0
    num_in_images = 0
    num_out_images = 0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    out_iterable = iter(train_loader_out)
    for e, in_set in enumerate(train_loader_in):

        try:
            out_set = next(out_iterable)
        except StopIteration:
            out_iterable = iter(train_loader_out)
            out_set = next(out_iterable)

        num_in_images += len(in_set[0])
        num_out_images += len(out_set[0])

        # data = in_set[0].to('cuda')
        # target = in_set[1].to('cuda')
        # x = net(data)
        # loss = F.cross_entropy(x, target)

        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.to('cuda'), target.to('cuda')

        # forward
        permutation_idx = torch.randperm(len(data))
        fake_target = torch.cat([target, torch.ones(len(out_set[0])).to('cuda') * -1], -1)
        binary_labels = torch.ones(len(data)).to('cuda')
        binary_labels[len(in_set[0]):] = 0
        x = net(data[permutation_idx])

        optimizer.zero_grad()

        # cross-entropy from softmax distribution to uniform distribution
        if args.add_class:
            target = torch.cat([target, torch.ones(len(out_set[0])).to('cuda').long() * (num_classes - 1)], -1)
            loss = F.cross_entropy(x, target)
        else:
            # breakpoint()
            loss = F.cross_entropy(x[binary_labels[permutation_idx].bool()], fake_target[permutation_idx][binary_labels[permutation_idx].bool()].long())
            Ec_out = torch.logsumexp(x[(1-binary_labels[permutation_idx]).bool()], dim=1) / args.T
            Ec_in = torch.logsumexp(x[binary_labels[permutation_idx].bool()], dim=1) / args.T


            input_for_lr = torch.cat((Ec_in, Ec_out), -1)
            criterion = torch.nn.CrossEntropyLoss()
            # breakpoint()
            output1 = logistic_regression(input_for_lr.reshape(-1,1)).to('cuda')
            energy_reg_loss = criterion(output1, binary_labels.long())


            loss += args.energy_weight * energy_reg_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        if e % 100 == 0:
            print('FIXED?',loss)
        # exponential moving average
        loss_energy_avg = loss_energy_avg * 0.8 + float(args.energy_weight * energy_reg_loss) * 0.2
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    print(scheduler.get_lr())
    print('loss energy is: ', loss_energy_avg)
    state['train_loss'] = loss_avg
    state['train_energy_loss'] = loss_energy_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    feat_all = []
    target_all = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cuda'), target.to('cuda')

            # forward
            output = net(data)
            # feat_all.append(feat)
            # target_all.append(target)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)
    # breakpoint()
    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)


save_info = save_info + "_slope_" + str(args.add_slope) + '_' + "weight_" + str(args.energy_weight)

# Make log directory
if not os.path.exists(os.path.join(args.save, 'test' + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                    '_' + save_info + '_training_results.csv')):
    os.makedirs(os.path.join(args.save, 'test' + calib_indicator + '_' + args.model + '_s' + str(args.seed) + 
                            '_' + save_info), exist_ok=True)

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                '_' + save_info + '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print(f"Beginning Training")

# Main loop
loss_min = 100
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train_permute()
    test()

    torch.save(net.state_dict(),
            os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                            '_' + save_info + '_epoch_' + str(epoch) + args.suffix + '.pt'))

    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                            '_' + save_info + '_epoch_' + str(epoch - 1) + args.suffix + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    print(f"Model saved at epoch {epoch} as {args.save + args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + '_' + save_info + '_epoch_' + str(epoch) + args.suffix + '.pt'}")

    # Show results
    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                    '_' + save_info + '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))


    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy']),
        flush=True
    )
