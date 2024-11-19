
from typing import Union
import os
import shutil

from torch import nn

import faiss as fs
import numpy as np

__all__ = [
    'ParallelType',
    'expanduservars',
    'archive_code',
    'WithStateDict'
]

ParallelType = Union[nn.DataParallel, nn.parallel.DistributedDataParallel]


class WithStateDict(nn.Module):
    """Wrapper to provide a `state_dict` method to a single tensor."""
    def __init__(self, **tensors):
        super().__init__()
        for name, value in tensors.items():
            self.register_buffer(name, value)


def expanduservars(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def archive_code(path: str) -> None:
    shutil.copy("nl-invs/params.yml", path)
    # Copy the current code to the output folder.
    os.system(f"git ls-files -z | xargs -0 tar -czf {os.path.join(path, 'code.tar.gz')}")


def knn_score(train_set, test_set, n_neighbours=2, train=False):
    """
    Calculates the KNN distance
    """
    index = fs.IndexFlatL2(train_set.shape[1])
    index.add(train_set)

    if train:
        D, _ = index.search(test_set, n_neighbours + 1)
        return np.sum(D[:, 1:], axis=1)
    else:
        D, _ = index.search(test_set, n_neighbours)
        return np.sum(D, axis=1)