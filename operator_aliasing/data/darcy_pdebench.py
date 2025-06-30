"""Simple Darcy Dataset Example."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
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
        num_samples_max: int = -1,
        transform: Compose = None,
    ):
        """Initialize data.

        :param filename: filename that contains the dataset
        :type filename: STR

        """
        self.transform = transform
        initial_step = 1
        reduced_resolution = 1
        reduced_batch = 1
        test_ratio = 0.1

        # Define path to files
        root_path = Path(Path(saved_folder).resolve()) / filename
        if filename[-2:] != 'h5':
            # print(".HDF5 file extension is assumed hereafter")

            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()

                ## data dim = [t, x1, ..., xd, v]
                _data = np.array(
                    f['tensor'], dtype=np.float32
                )  # batch, time, x,...
                four = 4
                if len(_data.shape) == four:  # 2D Darcy flow
                    # u: label
                    _data = _data[
                        ::reduced_batch,
                        :,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    # if _data.shape[-1]==1:  # if nt==1
                    #    _data = np.tile(_data, (1, 1, 1, 2))
                    self.data = _data
                    # nu: input
                    _data = np.array(
                        f['nu'], dtype=np.float32
                    )  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        None,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    self.data = np.concatenate([_data, self.data], axis=-1)
                    self.data = self.data[
                        :,
                        :,
                        :,
                        :,  # None
                    ]  # batch, x, y, t, ch

                    x = np.array(f['x-coordinate'], dtype=np.float32)
                    y = np.array(f['y-coordinate'], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    """
                    X, Y = torch.meshgrid(x, y, indexing='ij')
                    self.grid = torch.stack((X, Y), axis=-1)[
                        ::reduced_resolution, ::reduced_resolution
                    ]
                    """

        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if train:
            self.data = self.data[test_idx:num_samples_max]
        else:
            self.data = self.data[:test_idx]

        # Time steps used as initial conditions
        self.initial_step = initial_step

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
        unsqueeze_dim = 0
        if type(idx) is not int:
            unsqueeze_dim = 1
        sample = {
            'x': self.data[idx, :, :, 0].unsqueeze(unsqueeze_dim),
            'y': self.data[idx, ..., 1].unsqueeze(unsqueeze_dim),
            # self.grid,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
