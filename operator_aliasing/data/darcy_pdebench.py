"""Simple Darcy Dataset Example."""

from __future__ import annotations

import math
import typing
from pathlib import Path

import h5py
import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset
from torchvision.transforms import Compose

# https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py


class DarcyPDEBench(Dataset):
    """Darcy Dataset from PDE Bench."""

    def __init__(
        self,
        filename: str,
        saved_folder: str = '../data/',
        train: bool = True,
        transform: Compose = None,
        **kwargs: typing.Any,
    ):
        """Initialize data.

        :param filename: filename that contains the dataset
        :type filename: STR

        support img dimentions: 128, 64, 32, 16 (highest to lowest)
        """
        self.transform = transform
        # downsample data
        resolution_proportions = kwargs['resolution_proportions']
        four = 4
        assert len(resolution_proportions) == four, (
            'Only support 4 img_resolutions, see doc string.'
        )
        assert sum(resolution_proportions) == 1, (
            'All dataset proportions must sum to 1.'
        )
        self.rng = default_rng(seed=kwargs['seed'])
        self.batch_size = kwargs['batch_size']
        test_ratio = 0.1
        num_samples_max = -1

        # Define path to files
        self.model_inputs = []
        self.labels = []
        root_path = Path(Path(saved_folder).resolve()) / filename
        # preprocessing for train/test sets and
        # ...initializing/shuffling data indexes
        with h5py.File(root_path, 'r') as f:
            # num of data samples
            num_samples_max = f['tensor'].shape[0]

            # list of data idxs
            data_idx = np.arange(0, num_samples_max)
            # num of test samples
            test_idx = int(num_samples_max * test_ratio)
            if train:
                first_batch_idx = test_idx
                last_batch_idx = -1
                self.num_samples = num_samples_max - test_idx
            else:
                first_batch_idx = 0
                last_batch_idx = test_idx
                self.num_samples = test_idx
            print(f'{self.num_samples=}')
            # grab data indexs
            self.data_idxs = data_idx[first_batch_idx:last_batch_idx]
            # shuffle indexes
            self.rng.shuffle(self.data_idxs)

        for res_factor, ratio in enumerate(resolution_proportions):
            reduced_resolution = 2**res_factor
            with h5py.File(root_path, 'r') as f:
                # number of points in this resolution set
                res_idx = int(self.num_samples * ratio)
                # sort all indexes
                set_indexes = np.sort(self.data_idxs[:res_idx])
                # u: label
                label = np.array(
                    f['tensor'][
                        set_indexes,
                        :,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ],
                    dtype=np.float32,
                )

                # batch, time, x,...
                _data = np.array(f['nu'], dtype=np.float32)
                # nu: input
                model_input = _data[
                    set_indexes,
                    None,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ]

                self.model_inputs.append(torch.tensor(model_input))
                self.labels.append(torch.tensor(label))

                # remove already used indexes
                self.data_idxs = self.data_idxs[res_idx:]

    def __len__(self) -> int:
        """Returns len of dataset.

        Recall this is a pre-batched dataset, so we return
        number of batches.
        """
        return math.ceil(self.num_samples / self.batch_size)

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        """Get single sample at idx."""
        # iterate through all resoulution sets to find batch
        for _set_idx, res_set in enumerate(self.model_inputs):
            num_batches_in_set = math.ceil(len(res_set) / self.batch_size)
            if batch_idx >= num_batches_in_set:
                batch_idx -= num_batches_in_set
            else:
                item_idx = int(batch_idx * self.batch_size)
                set_idx = _set_idx
                break

        # return whole batch, not just single datapoint
        sample = {
            'x': self.model_inputs[set_idx][
                item_idx : item_idx + self.batch_size
            ],
            'y': self.labels[set_idx][item_idx : item_idx + self.batch_size],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
