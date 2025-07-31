"""(Compressible) Navier Stokes Dataset Example."""

# https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py
from __future__ import annotations

import typing
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class NSPDEBench(Dataset):
    """(Compressible) Navier Stokes Dataset from PDE Bench."""

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
        **kwargs: typing.Any,
    ):
        """Initialize data.

        :param filename: filename that contains the dataset
        :type filename: STR

        """
        self.transform = transform
        self.initial_step = initial_step
        # downsample data
        img_size = kwargs['img_size']
        spatial_dim = 128  # from the real data TODO(MS): check for real data
        if spatial_dim % img_size != 0:
            raise Exception(f"""Desired img_size should
                be a factor of the data's {spatial_dim=}
                """)
        reduced_resolution = spatial_dim // img_size
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
                    print(f'{_data.shape=}')
                    # batch, time, x,...

                    # density
                    density = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    print('loaded density')
                    # pressure
                    _data = np.array(
                        f['pressure'], dtype=np.float32
                    )  # batch, time, x,...
                    pressure = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    print('loaded pressure')
                    # Vx
                    _data = np.array(
                        f['Vx'], dtype=np.float32
                    )  # batch, time, x,...
                    vx = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    print('loaded Vx')
                    # Vy
                    _data = np.array(
                        f['Vy'], dtype=np.float32
                    )  # batch, time, x,...
                    vy = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    print('loaded Vy')

                    self.data = np.stack([density, pressure, vx, vy], axis=2)

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
