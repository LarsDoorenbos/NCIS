
import numpy as np

import glob
import os

import torchvision




dataset = 'in100'


files = glob.glob(('embeddings/*' + dataset + '*.npy'))
sorted_files = sorted(files, key=lambda x: int(x.split('_')[-2]))
for file in sorted_files:
    print(file)
    if 'embeddings' in locals():
        embeddings = np.concatenate((embeddings, np.load(file)), axis=0)
    else:
        embeddings = np.load(file)

print(embeddings.shape)


labels = []

if dataset == 'c10':
    train_data = torchvision.datasets.CIFAR10("../data/", train=True)
elif dataset == 'c100':
    train_data = torchvision.datasets.CIFAR100("../data/", train=True)
elif dataset == 'in100':
    traindir = os.path.join('/storage/workspaces/artorg_aimi/ws_00000/lars', 'imgnet_100_train')
    train_data = torchvision.datasets.ImageFolder(traindir)

for i in range(len(train_data)):
    labels.append(train_data[i][1])

labels = np.array(labels)
num_classes = np.unique(labels).shape[0]

if dataset == 'c10' or dataset == 'c100':
    # Group embeddings by class label
    grouped_embeddings = np.zeros((num_classes, int(len(labels) / num_classes), embeddings.shape[1]))

    for i in range(num_classes):
        grouped_embeddings[i] = embeddings[labels == i]
else:
    grouped_embeddings = embeddings

    # Save labels
    np.save(dataset + '_labels.npy', labels)

print(grouped_embeddings.shape, len(labels))

# Save embeddings
np.save(dataset + '_embeddings.npy', grouped_embeddings)

print(f"Saved to {dataset}_embeddings.npy")