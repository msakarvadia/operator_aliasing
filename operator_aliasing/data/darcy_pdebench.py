"""Simple Darcy Dataset Example."""

from __future__ import annotations

import typing
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
        transform: Compose = None,
        **kwargs: typing.Any,
    ):
        """Initialize data.

        :param filename: filename that contains the dataset
        :type filename: STR

        """
        self.transform = transform
        initial_step = 1
        # downsample data
        img_size = kwargs['img_size']
        spatial_dim = 128  # from the real data TODO(MS): check for real data
        if spatial_dim % img_size != 0:
            raise Exception(f"""Desired img_size should
                be a factor of the data's {spatial_dim=}
                """)
        reduced_resolution = spatial_dim // img_size
        reduced_batch = 1
        test_ratio = 0.1
        num_samples_max = -1

        # Define path to files
        root_path = Path(Path(saved_folder).resolve()) / filename
        if filename[-2:] != 'h5':
            # print(".HDF5 file extension is assumed hereafter")

            with h5py.File(root_path, 'r') as f:
                num_samples_max = f['tensor'].shape[0]
                test_idx = int(num_samples_max * test_ratio)
                if train:
                    first_batch_idx = test_idx
                    last_batch_idx = -1
                else:
                    first_batch_idx = 0
                    last_batch_idx = test_idx

                # u: label
                label = np.array(
                    f['tensor'][
                        first_batch_idx:last_batch_idx:reduced_batch,
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
                    first_batch_idx:last_batch_idx:reduced_batch,
                    None,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ]

                self.model_input = torch.tensor(model_input)
                self.label = torch.tensor(label)

        # Time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self) -> int:
        """Returns len of dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get single sample at idx."""
        sample = {
            'x': self.model_input[idx],
            'y': self.label[idx],
            # self.grid,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
