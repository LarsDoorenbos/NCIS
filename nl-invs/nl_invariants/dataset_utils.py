
from typing import (Generic, TypeVar, Callable, Sequence,
                    Pattern, Union, List, Dict, Any, cast)

import h5py
import imageio

from torch.utils.data import Dataset

Tin = TypeVar('Tin')
Tout = TypeVar('Tout')


class EmptyDataset(Dataset):

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError('`EmptyDataset` is empty')


class H5Dataset(Dataset):

    def __init__(self, h5file: str, dataset_key: str):
        self.h5file = h5py.File(h5file, 'r')
        self.dataset = self.h5file[dataset_key]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class NumpyDataset(Dataset, Generic[Tin, Tout]):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return x, y

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels


class FileListDataset(Dataset, Generic[Tin, Tout]):

    def __init__(self,
                 file_list: Sequence[Tin],
                 loader: Callable[[Tin], Tout] = imageio.imread
                 ) -> None:
        self.loader = loader
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tout:
        return self.loader(self.file_list[idx])


class FileListReDataset(FileListDataset, Generic[Tin, Tout]):

    def __init__(self,
                 file_list: Sequence[Tin],
                 regexp: Pattern,
                 labels: Union[List[str], Dict[str, Any]],
                 loader: Callable[[Tin], Tout] = imageio.imread
                 ) -> None:

        super().__init__(file_list, self._loader)
        self.regexp = regexp
        self.base_loader = loader

        if isinstance(labels, list):
            labels = {lbl: i for i, lbl in enumerate(labels)}
        self.labels = cast(Dict[str, Any], labels)

    def _loader(self, filename: Tin):

        x = self.base_loader(filename)

        match = self.regexp.match(filename)
        if match is None:
            raise ValueError(
                "could not find a match with file name `{}`".format(filename)
            )
        grp = match.group(1)
        label = self.labels[grp]

        return x, label


class TransformedDataset(Dataset, Generic[Tin, Tout]):

    def __init__(self,
                 source_dataset: Dataset,
                 transform_func: Callable[..., Tout]
                 ) -> None:
        self.source_dataset = source_dataset
        self.transform_func = transform_func

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx: int) -> Tout:

        value = self.source_dataset[idx]

        if isinstance(value, tuple):
            return self.transform_func(*value)

        return self.transform_func(value)
