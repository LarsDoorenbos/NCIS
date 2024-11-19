
import numpy as np

import torch
import torch.utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def training_dataset(preprocessing, train_fraction):
    
    data = np.load('c100_embeddings.npy')

    # 100x500x768, ordered by class
    data = data.reshape((50000, 768))

    labels = [i * np.ones(500) for i in range(100)]
    labels = np.concatenate(labels)

    if preprocessing == 'center':
        data = data - np.mean(data, axis=0, keepdims=True)

    if train_fraction < 1:
        n = np.random.choice(len(data), int(train_fraction * len(data)), replace=False)
        data = data[n]
        labels = labels[n]

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))

    return dataset