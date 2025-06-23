"""Utility functions for fetching and managing data."""

from __future__ import annotations

import typing

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from operator_aliasing.data.random_data import RandomData
from operator_aliasing.data.transforms import DownSample
from operator_aliasing.data.transforms import LowpassFilter2D

from ..utils import seed_everything
from ..utils import seed_worker


def get_dataset(
    **data_args: typing.Any,
) -> Dataset:
    """Get specific dataset w/ transform."""
    dataset_name = data_args['dataset_name']
    filter_lim = data_args['filter_lim']
    img_size = data_args['img_size']
    downsample_dim = data_args['downsample_dim']
    train = data_args['train']
    # Handle data transformations
    data_transforms = transforms.Compose(
        [LowpassFilter2D(filter_lim, img_size), DownSample(downsample_dim)]
    )

    # grab specific dataset
    if dataset_name == 'random':
        dataset = RandomData(
            n_train=100, train=train, transform=data_transforms
        )
    return dataset


def get_data(
    **data_args: typing.Any,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Get data w/ args."""
    batch_size = data_args['batch_size']
    seed = data_args['seed']

    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_kwargs = {
        'dataset_name': 'random',
        'filter_lim': 3,
        'img_size': 16,
        'downsample_dim': -1,
        'train': True,
    }
    train_dataset = get_dataset(**train_kwargs)
    test_kwargs = {
        'dataset_name': 'random',
        'filter_lim': 3,
        'img_size': 16,
        'downsample_dim': -1,
        'train': False,
    }
    test_dataset = get_dataset(**test_kwargs)
    """
    filter_lim = 3
    img_size = 16
    downsample_dim = -1
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
    """

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
