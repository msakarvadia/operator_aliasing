"""Utility functions for fetching and managing data."""

from __future__ import annotations

import typing

import torch
from torch.utils.data import DataLoader

from operator_aliasing.data.darcy import get_darcy_data
from operator_aliasing.data.random_data import get_random_data

from ..utils import seed_everything
from ..utils import seed_worker


def get_data(
    **data_args: typing.Any,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Get data w/ args."""
    batch_size = data_args['batch_size']
    seed = data_args['seed']

    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset, test_dataset = get_random_data(100)
    train_dataset, test_datasets = get_darcy_data()

    training_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    testing_loaders = {}
    for k, test_dataset in test_datasets.items():
        testing_loaders[k] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
    return (training_loader, testing_loaders)
    # return (training_loader, {'test': testing_loader})
