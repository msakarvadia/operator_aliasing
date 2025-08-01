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
        spatial_dim = 510  # from the real data TODO(MS): check for real data
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
            # force curl: bs x X x Y
            num_samples_max = f['force_curl'].shape[0]
            test_idx = int(num_samples_max * test_ratio)
            if train:
                first_batch_idx = test_idx
                last_batch_idx = -1
            else:
                first_batch_idx = 0
                last_batch_idx = test_idx
            print(f'{num_samples_max=}')
            self.force_curl = np.array(
                f['force_curl'][
                    first_batch_idx:last_batch_idx:reduced_batch,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ],
                dtype=np.float32,
            )
            print(f'loaded force curl {self.force_curl.shape=}')
            # vorticity
            self.vorticity = torch.tensor(
                f['vorticity'][
                    first_batch_idx:last_batch_idx:reduced_batch,
                    ::reduced_resolution_t,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ],
                dtype=torch.float32,
            )  # batch, time, x, y
            print(f'loaded vorticity {self.vorticity.shape=}')
            # Vx
            self.vx = torch.tensor(
                f['Vx'][
                    first_batch_idx:last_batch_idx:reduced_batch,
                    ::reduced_resolution_t,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ],
                dtype=torch.float32,
            )  # batch, time, x,...
            print('loaded Vx')
            # Vy
            self.vy = torch.tensor(
                f['Vy'][
                    first_batch_idx:last_batch_idx:reduced_batch,
                    ::reduced_resolution_t,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ],
                dtype=torch.float32,
            )  # batch, time, x,...
            print('loaded Vy')

        # batch, time, channel, x, y
        self.vorticity = self.vorticity[:, :, None, :, :]

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
