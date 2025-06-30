"""Simple Darcy Dataset Example."""

from __future__ import annotations

import torch
from neuralop.data.datasets.darcy import DarcyDataset
from neuralop.utils import get_project_root
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import math as mt
from pathlib import Path
import numpy as np

import h5py
import torch
from torch.utils.data import Dataset

# https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py

class DarcyPDEBench(Dataset):
    def __init__(
        self,
        filename,
        initial_step=1,
        saved_folder='../data/',
        reduced_resolution=1,
        reduced_resolution_t=1,
        reduced_batch=1,
        #if_test=False,
        train: bool = True,
        test_ratio=0.1,
        num_samples_max=-1,
        transform: Compose = None,
    ):
        """:param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        self.transform = transform

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
                if len(_data.shape) == 3:  # 1D
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # batch, x, t, ch

                    self.grid = np.array(
                        f['x-coordinate'], dtype=np.float32
                    )
                    self.grid = torch.tensor(
                        self.grid[::reduced_resolution], dtype=torch.float
                    ).unsqueeze(-1)
                if len(_data.shape) == 4:  # 2D Darcy flow
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
                        :, :, :, :,  #None
                    ]  # batch, x, y, t, ch

                    x = np.array(f['x-coordinate'], dtype=np.float32)
                    y = np.array(f['y-coordinate'], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    X, Y = torch.meshgrid(x, y, indexing='ij')
                    self.grid = torch.stack((X, Y), axis=-1)[
                        ::reduced_resolution, ::reduced_resolution
                    ]


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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        unsqueeze_dim = 0
        if type(idx) != int:
            unsqueeze_dim = 1
        sample = {
            "x": self.data[idx, :, :, 0].unsqueeze(unsqueeze_dim),
            "y": self.data[idx, ..., 1].unsqueeze(unsqueeze_dim),
            #self.grid,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
