"""Small random dataset for testing purposes."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torchvision.transforms import Compose


def get_random_data(
    n_train: int = 100,
) -> tuple[TensorDataset, dict[str, TensorDataset]]:
    """Small random dataset for testing purposes."""
    # NOTE(MS): depricating support for random data
    x_data = torch.rand((256, 1, 128, 128)).type(torch.float32)
    y_data = torch.ones((256, 1, 128, 128)).type(torch.float32)

    input_function_train = x_data[:n_train, :]
    output_function_train = y_data[:n_train, :]
    input_function_test = x_data[n_train:, :]
    output_function_test = y_data[n_train:, :]

    train_data = TensorDataset(input_function_train, output_function_train)
    test_data = TensorDataset(input_function_test, output_function_test)

    return train_data, {'test': test_data}


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
