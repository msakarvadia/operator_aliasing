"""Small random dataset for testing purposes."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset


def get_random_data(
    n_train: int = 100,
) -> tuple[TensorDataset, dict[str, TensorDataset]]:
    """Small random dataset for testing purposes."""
    x_data = torch.rand((256, 1, 128, 128)).type(torch.float32)
    y_data = torch.ones((256, 1, 128, 128)).type(torch.float32)

    input_function_train = x_data[:n_train, :]
    output_function_train = y_data[:n_train, :]
    input_function_test = x_data[n_train:, :]
    output_function_test = y_data[n_train:, :]

    train_data = TensorDataset(input_function_train, output_function_train)
    test_data = TensorDataset(input_function_test, output_function_test)

    return train_data, {'test': test_data}
