"""(Incompressible) Navier Stokes Dataset Example."""

# https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py
from __future__ import annotations

import typing
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class IncompNSPDEBench(Dataset):
    """(Incompressible) Navier Stokes Dataset from PDE Bench.

    Generate this dataset 1st by downloading Incompressible NS
    frome PDEBench and then using `preprocess_ncom_ns.py` to
    preprocess it into `full_data_merge.h5`.
    """

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
        spatial_dim = 512  # from the real data TODO(MS): check for real data
        if spatial_dim % img_size != 0:
            raise Exception(f"""Desired img_size should
                be a factor of the data's {spatial_dim=}
                """)
        reduced_resolution = spatial_dim // img_size
        reduced_resolution_t = (
            1  # NOTE(MS): already filtered time in preprocessing
        )
        reduced_batch = 1
        test_ratio = 0.1
        num_samples_max = -1

        # Define path to files
        root_path = Path(Path(saved_folder).resolve()) / filename
        with h5py.File(root_path, 'r') as f:
            _data = np.array(f['force_curl'], dtype=np.float32)
            # force curl: bs x X x Y
            self.force_curl = _data[
                ::reduced_batch,
                ::reduced_resolution,
                ::reduced_resolution,
            ]
            print(f'loaded force curl {self.force_curl.shape=}')
            # vorticity
            _data = np.array(
                f['vorticity'], dtype=np.float32
            )  # batch, time, x, y
            self.vorticity = _data[
                ::reduced_batch,
                ::reduced_resolution_t,
                ::reduced_resolution,
                ::reduced_resolution,
            ]
            print(f'loaded vorticity {self.vorticity.shape=}')
            # Vx
            _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
            self.vx = _data[
                ::reduced_batch,
                ::reduced_resolution_t,
                ::reduced_resolution,
                ::reduced_resolution,
            ]
            print('loaded Vx')
            # Vy
            _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
            self.vy = _data[
                ::reduced_batch,
                ::reduced_resolution_t,
                ::reduced_resolution,
                ::reduced_resolution,
            ]
            print('loaded Vy')

        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.vorticity.shape[0]

        test_idx = int(num_samples_max * test_ratio)

        # batch, time, channel, x, y
        self.vorticity = self.vorticity[:, :, None, :, :]

        if train:
            self.vorticity = self.vorticity[test_idx:num_samples_max]
            self.force_curl = self.force_curl[test_idx:num_samples_max]
            self.vx = self.vx[test_idx:num_samples_max]
            self.vy = self.vy[test_idx:num_samples_max]
        else:
            self.vorticity = self.vorticity[:test_idx]
            self.force_curl = self.force_curl[:test_idx]
            self.vx = self.vx[:test_idx]
            self.vy = self.vy[:test_idx]

        self.vorticity = (
            self.vorticity
            if torch.is_tensor(self.vorticity)
            else torch.tensor(self.vorticity)
        )

        self.force_curl = (
            self.force_curl
            if torch.is_tensor(self.force_curl)
            else torch.tensor(self.force_curl)
        )

        self.vx = (
            self.vx if torch.is_tensor(self.vx) else torch.tensor(self.vx)
        )

        self.vy = (
            self.vy if torch.is_tensor(self.vy) else torch.tensor(self.vy)
        )

    def __len__(self) -> int:
        """Returns len of dataset."""
        return len(self.vorticity)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get single sample at idx."""
        sample = {
            'x': self.vorticity[idx, : self.initial_step, ...],
            'y': self.vorticity[idx, ...],
            'Vx': self.vx[idx, ...],
            'Vy': self.vy[idx, ...],
            'force_curl': self.force_curl[idx, ...],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
