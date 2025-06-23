"""Utility functions for fetching and managing data."""

from __future__ import annotations

import typing

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from operator_aliasing.data.random_data import RandomData
from operator_aliasing.data.transforms import DownSample
from operator_aliasing.data.transforms import LowpassFilter2D

from ..utils import seed_everything
from ..utils import seed_worker


def get_data(
    **data_args: typing.Any,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Get data w/ args."""
    batch_size = data_args['batch_size']
    seed = data_args['seed']
    filter_lim = 3
    img_size = 16
    downsample_dim = -1

    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Handle data transformations
    data_transforms = transforms.Compose(
        [LowpassFilter2D(filter_lim, img_size), DownSample(downsample_dim)]
    )

    # train_dataset, test_datasets = get_random_data(100)
    train_dataset = RandomData(
        n_train=100, train=True, transform=data_transforms
    )
    test_dataset = RandomData(
        n_train=100, train=False, transform=data_transforms
    )
    # train_dataset, test_datasets = get_darcy_data(data_transforms)

    # NOTE(MS): how to handle different transforms for testing?
    test_datasets = {'test': test_dataset}

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
