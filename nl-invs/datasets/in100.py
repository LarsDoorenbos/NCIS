
import numpy as np

import torch
import torch.utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def training_dataset(preprocessing):
    
    data = np.load('in100_embeddings.npy')
    labels = np.load('in100_labels.npy')

    if preprocessing == 'center':
        data = data - np.mean(data, axis=0, keepdims=True)

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))

    return dataset