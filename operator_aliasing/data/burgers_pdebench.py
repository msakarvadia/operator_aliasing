"""Burgers Dataset."""

# https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py
# https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class BurgersPDEBench(Dataset):
    """Burger Dataset from PDE Bench."""

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
        """Initialize data."""
        self.transform = transform
        reduced_resolution = 1
        reduced_resolution_t = 2
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

                ## data dim = [t, x1, ..., xd, v]
                _data = np.array(
                    f['tensor'], dtype=np.float32
                )  # batch, time, x,...
                _data = _data[
                    ::reduced_batch,
                    ::reduced_resolution_t,
                    ::reduced_resolution,
                ]
                # batch, time, channel, x,
                self.data = _data[:, :, None, :]

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
        sample = {
            'x': self.data[idx, : self.initial_step, ...],
            'y': self.data[idx, ...],
            # self.grid,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
