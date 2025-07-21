"""Navier Stokes Dataset Example."""

# https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py
from __future__ import annotations

import math as mt
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class NSPDEBench(Dataset):
    """Navier Stokes Dataset from PDE Bench."""

    def __init__(
        self,
        filename: str,
        initial_step: int = 10,
        saved_folder: str = '../data/',
        # reduced_resolution=1,
        # reduced_resolution_t=1,
        # reduced_batch=1,
        train: bool = False,
        # test_ratio=0.1,
        # num_samples_max=-1,
        transform: Compose = None,
    ):
        """Initialize data.

        :param filename: filename that contains the dataset
        :type filename: STR

        """
        self.transform = transform
        self.initial_step = initial_step
        reduced_resolution = 1
        reduced_resolution_t = 1
        reduced_batch = 1
        test_ratio = 0.1
        num_samples_max = -1

        # Define path to files
        root_path = Path(Path(saved_folder).resolve()) / filename
        if filename[-2:] != 'h5':
            # print(".HDF5 file extension is assumed hereafter")

            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)
                    # batch, time, x,...
                    idx_cfd = _data.shape

                    self.data = np.zeros(
                        [
                            idx_cfd[0] // reduced_batch,
                            idx_cfd[2] // reduced_resolution,
                            idx_cfd[3] // reduced_resolution,
                            mt.ceil(idx_cfd[1] / reduced_resolution_t),
                            4,
                        ],
                        dtype=np.float32,
                    )
                    # density
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 0] = _data  # batch, x, t, ch
                    print('loaded density')
                    # pressure
                    _data = np.array(
                        f['pressure'], dtype=np.float32
                    )  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 1] = _data  # batch, x, t, ch
                    print('loaded pressure')
                    # Vx
                    _data = np.array(
                        f['Vx'], dtype=np.float32
                    )  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 2] = _data  # batch, x, t, ch
                    print('loaded Vx')
                    # Vy
                    _data = np.array(
                        f['Vy'], dtype=np.float32
                    )  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))

                    self.data[..., 3] = _data  # batch, x, t, ch
                    print('loaded Vy')

                    # batch, time, channel, x, y
                    self.data = np.transpose(self.data, (0, 3, 4, 1, 2))

        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if train:
            self.data = self.data[test_idx:num_samples_max]
        else:
            self.data = self.data[:test_idx]

        self.data = (
            self.data
            if torch.is_tensor(self.data)
            else torch.tensor(self.data)
        )

    def __len__(self) -> int:
        """Returns len of dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get single sample at idx."""
        sample = {
            'x': self.data[idx, : self.initial_step, ...],
            'y': self.data[idx, ...],
            # self.grid,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
