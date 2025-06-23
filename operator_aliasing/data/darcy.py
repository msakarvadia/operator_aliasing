"""Simple Darcy Dataset Example."""

from __future__ import annotations

import torch
from neuralop.data.datasets.darcy import DarcyDataset
from neuralop.utils import get_project_root
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class DarcyData(Dataset):
    """Small random dataset for testing purposes."""

    def __init__(
        self,
        n_train: int = 100,
        transform: Compose = None,
        train: bool = True,
        img_size: int = 16,
    ) -> None:
        """Initialize dataset."""
        self.n_train = n_train
        self.transform = transform
        self.train = train

        root_dir = get_project_root() / 'neuralop/data/datasets/data'
        data = DarcyDataset(
            root_dir=root_dir,
            n_train=n_train,
            n_tests=[32, 32, 32, 32],
            batch_size=16,
            test_batch_sizes=[16, 16, 16, 16],
            train_resolution=img_size,  # change res to download different data
            test_resolutions=[img_size],
        )

        self.data = data.test_dbs[img_size]
        if self.train:
            self.data = data.train_db

    def __len__(self) -> int:
        """Return len of dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get single sample at idx."""
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
