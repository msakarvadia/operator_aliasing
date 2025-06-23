"""Small random dataset for testing purposes."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class RandomData(Dataset):
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

        x_data = torch.rand((256, 1, img_size, img_size)).type(torch.float32)
        y_data = torch.ones((256, 1, img_size, img_size)).type(torch.float32)
        self.input_function = x_data[n_train:, :]
        self.output_function = y_data[n_train:, :]
        if self.train:
            self.input_function = x_data[:n_train, :]
            self.output_function = y_data[:n_train, :]

    def __len__(self) -> int:
        """Return len of dataset."""
        return self.input_function.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get single sample at idx."""
        model_input = self.input_function[idx]
        label = self.output_function[idx]
        sample = {'x': model_input, 'y': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
