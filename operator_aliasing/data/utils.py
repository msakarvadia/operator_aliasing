"""Utility functions for fetching and managing data."""

from __future__ import annotations

import typing

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def get_data(**data_args: typing.Any) -> tuple[DataLoader, DataLoader]:
    """Get data w/ args."""
    batch_size = data_args['batch_size']

    n_train = 100  # number of training samples

    x_data = torch.rand((256, 1, 128, 128)).type(torch.float32)
    y_data = torch.ones((256, 1, 128, 128)).type(torch.float32)

    input_function_train = x_data[:n_train, :]
    output_function_train = y_data[:n_train, :]
    input_function_test = x_data[n_train:, :]
    output_function_test = y_data[n_train:, :]

    training_loader = DataLoader(
        TensorDataset(input_function_train, output_function_train),
        batch_size=batch_size,
        shuffle=True,
    )
    testing_loader = DataLoader(
        TensorDataset(input_function_test, output_function_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return (training_loader, testing_loader)
