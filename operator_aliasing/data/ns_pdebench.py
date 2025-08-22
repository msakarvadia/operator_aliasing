"""(Compressible) Navier Stokes Dataset Example."""

# https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py
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


class NSPDEBench(Dataset):
    """(Compressible) Navier Stokes Dataset from PDE Bench."""

    def __init__(
        self,
        filename: str,
        initial_step: int = 10,
        saved_folder: str = '../data/',
        train: bool = False,
        transform: Compose = None,
        **kwargs: typing.Any,
    ):
        """Initialize data.

        :param filename: filename that contains the dataset
        :type filename: STR

        support img dimentions: 128, 64, 32, 16 (highest to lowest)
                             OR 512, 256, 128, 64, (highest to lowest)

        spatial dim just depends on the original simulation res
        """
        self.transform = transform
        self.initial_step = initial_step
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
        # NOTE(MS): already filtered time in preprocessing
        self.reduced_resolution_t = 1

        self.root_path = Path(Path(saved_folder).resolve()) / filename
        with h5py.File(self.root_path, 'r') as f:
            # num of data samples
            num_samples_max = f['density'].shape[0]

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

        # Define path to files
        self.index_sets = []
        for _res_factor, ratio in enumerate(resolution_proportions):
            # number of points in this resolution set
            res_idx = int(self.num_samples * ratio)
            # sort all indexes
            set_indexes = np.sort(self.data_idxs[:res_idx])
            self.index_sets.append(set_indexes)
            # remove already used indexes
            self.data_idxs = self.data_idxs[res_idx:]

    def __len__(self) -> int:
        """Returns len of dataset.

        Recall this is a pre-batched dataset, so we return
        number of batches.
        """
        total_batches = 0
        for _set_idx, res_set in enumerate(self.index_sets):
            num_batches_in_set = math.ceil(len(res_set) / self.batch_size)
            total_batches += num_batches_in_set
        return total_batches

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        """Get single batch."""
        # iterate through all resoulution sets to find batch
        for _set_idx, res_set in enumerate(self.index_sets):
            num_batches_in_set = math.ceil(len(res_set) / self.batch_size)
            if batch_idx >= num_batches_in_set:
                batch_idx -= num_batches_in_set
            else:
                item_idx = int(batch_idx * self.batch_size)
                set_idx = _set_idx
                break

        reduced_resolution = 2**set_idx
        set_indexes = self.index_sets[set_idx][
            item_idx : item_idx + self.batch_size
        ]
        with h5py.File(self.root_path, 'r') as f:
            density = np.array(
                f['density'][
                    set_indexes,
                    :: self.reduced_resolution_t,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ],
                dtype=np.float32,
            )
            # batch, time, x,...
            # pressure
            pressure = np.array(
                f['pressure'][
                    set_indexes,
                    :: self.reduced_resolution_t,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ],
                dtype=np.float32,
            )  # batch, time, x,...
            # Vx
            vx = np.array(
                f['Vx'][
                    set_indexes,
                    :: self.reduced_resolution_t,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ],
                dtype=np.float32,
            )  # batch, time, x,...
            # Vy
            vy = np.array(
                f['Vy'][
                    set_indexes,
                    :: self.reduced_resolution_t,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ],
                dtype=np.float32,
            )  # batch, time, x,...

            self.data_set = torch.tensor(
                np.stack([density, pressure, vx, vy], axis=2)
            )

        sample = {
            'x': self.data_set[
                :,
                : self.initial_step,
                ...,
            ],
            'y': self.data_set,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
