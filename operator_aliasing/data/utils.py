"""Utility functions for fetching and managing data."""

from __future__ import annotations

import typing

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from operator_aliasing.data.darcy import DarcyData
from operator_aliasing.data.darcy_pdebench import DarcyPDEBench
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

    # NOTE(MS): change filter size if downsampling
    # filter_size = img_size
    # if downsample_dim != -1:
    #    filter_size = downsample_dim

    # Handle data transformations
    data_transforms = transforms.Compose(
        # NOTE (MS): downsample before filter
        # [DownSample(downsample_dim),LowpassFilter2D(filter_lim, filter_size)]
        [LowpassFilter2D(filter_lim, img_size), DownSample(downsample_dim)]
    )

    # grab specific dataset
    if dataset_name == 'random':
        data_class = RandomData
        dataset = data_class(
            n_train=100,
            train=train,
            transform=data_transforms,
            img_size=img_size,
        )
    if dataset_name == 'darcy':
        data_class = DarcyData
        dataset = data_class(
            n_train=1000,
            train=train,
            transform=data_transforms,
            img_size=img_size,
        )
    if dataset_name == 'darcy_pdebench':
        dataset = DarcyPDEBench(
            filename='2D_DarcyFlow_beta0.01_Train.hdf5',
            # initial_step=1,
            saved_folder='/pscratch/sd/m/mansisak/PDEBench/pdebench_data/2D/DarcyFlow/',
            # reduced_resolution=1,
            # reduced_resolution_t=1,
            # reduced_batch=1,
            train=train,
            # test_ratio=0.1,
            num_samples_max=-1,
            transform=data_transforms,
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

    # set train specific kwarg
    data_args['train'] = True
    train_dataset = get_dataset(**data_args)

    test_datasets = {}
    """
    for downsample in [-1, 8, 11]:
        for lim in [-1, 5]:
            # do not test on downsampled unfiltered data
            if lim == -1 and downsample != -1:
                continue
            test_kwargs = data_args
            # set test specific kwargs
            test_kwargs['train'] = False
            test_kwargs['filter_lim'] = lim
            test_kwargs['downsample_dim'] = downsample

            test_dataset = get_dataset(**test_kwargs)
            test_datasets[f'test_{lim=}_{downsample=}'] = test_dataset
    """
    test_kwargs = data_args
    test_kwargs['train'] = False
    test_dataset = get_dataset(**test_kwargs)
    test_datasets['test'] = test_dataset

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
