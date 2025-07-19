"""Small random dataset for fluid flow testing purposes."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class RandomFluidData(Dataset):
    """Small random dataset for testing purposes.

    data contains a time component

    data contains multiple channels
    """

    def __init__(
        self,
        n_train: int = 100,
        transform: Compose = None,
        train: bool = True,
        img_size: int = 16,
        initial_steps: int = 10,
    ) -> None:
        """Initialize dataset."""
        self.n_train = n_train
        self.transform = transform
        self.train = train
        self.img_size = img_size

        self.init_steps = initial_steps
        self.total_steps = 21
        self.num_channels = 4

        x_data = torch.rand(
            (256, self.total_steps, self.num_channels, img_size, img_size)
        ).type(torch.float32)
        self.input_function = x_data[n_train:, :]
        if self.train:
            self.input_function = x_data[:n_train, :]

    def __len__(self) -> int:
        """Return len of dataset."""
        return self.input_function.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get single sample at idx."""
        model_input = self.input_function[idx, : self.init_steps, ...]
        label = self.input_function[idx, ...]

        model_input = torch.reshape(
            model_input,
            (
                self.init_steps * self.num_channels,
                self.img_size,
                self.img_size,
            ),
        )
        # TODO(MS): make Y-data in terms of X-data
        sample = {'x': model_input, 'y': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
