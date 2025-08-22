"""Utility functions for fetching and managing data."""

from __future__ import annotations

import typing

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from operator_aliasing.data.burgers_pdebench import BurgersPDEBench
from operator_aliasing.data.darcy import DarcyData
from operator_aliasing.data.darcy_pdebench import DarcyPDEBench
from operator_aliasing.data.incomp_ns_pdebench import IncompNSPDEBench
from operator_aliasing.data.ns_pdebench import NSPDEBench
from operator_aliasing.data.random_data import RandomData
from operator_aliasing.data.random_fluid_data import RandomFluidData
from operator_aliasing.data.transforms import DownSample
from operator_aliasing.data.transforms import LowpassFilter

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
    initial_steps = data_args['initial_steps']
    model_name = data_args['model_name']
    darcy_forcing_term = data_args['darcy_forcing_term']
    burger_viscosity = data_args['burger_viscosity']
    batch_size = data_args['batch_size']
    seed = data_args['seed']
    resolution_ratios = data_args['resolution_ratios']
    print(f'{resolution_ratios=}')

    n_spatial_dims = 2
    if '1D' in model_name:
        n_spatial_dims = 1

    # NOTE(MS): change filter size if downsampling
    # filter_size = img_size
    # if downsample_dim != -1:
    #    filter_size = downsample_dim

    # Handle data transformations
    data_transforms = transforms.Compose(
        # NOTE (MS): downsample before filter
        # [DownSample(downsample_dim),LowpassFilter2D(filter_lim, filter_size)]
        [
            # TODO(MS): need to figure out img size for low pass filter!!
            LowpassFilter(filter_lim, n_spatial_dims),
            DownSample(downsample_dim, n_spatial_dims),
        ]
    )

    # grab specific dataset
    if dataset_name == 'random_fluid':
        data_class: typing.Any = RandomFluidData
        dataset = data_class(
            n_train=100,
            train=train,
            # TODO(MS): add in compatible data transforms
            transform=data_transforms,
            # img_size=img_size,
            initial_steps=initial_steps,
            n_spatial_dims=n_spatial_dims,
        )
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
            filename=f'2D_DarcyFlow_beta{darcy_forcing_term}_Train.hdf5',
            saved_folder='/pscratch/sd/m/mansisak/PDEBench/pdebench_data/2D/DarcyFlow/',
            train=train,
            num_samples_max=-1,
            transform=data_transforms,
            batch_size=batch_size,
            resolution_proportions=resolution_ratios,
            seed=seed,
        )
    if dataset_name == 'burgers_pdebench':
        dataset = BurgersPDEBench(
            filename=f'1D_Burgers_Sols_Nu{burger_viscosity}.hdf5',
            initial_step=initial_steps,
            saved_folder='/pscratch/sd/m/mansisak/PDEBench/pdebench_data/1D/Burgers/Train/',
            train=train,
            transform=data_transforms,
            batch_size=batch_size,
            resolution_proportions=resolution_ratios,
            seed=seed,
        )
    if dataset_name == 'ns_pdebench':
        param = data_args['comp_ns_params']
        dataset = NSPDEBench(
            filename=(
                f'2D_CFD_{param[0]}_M{param[1]}_'
                f'Eta{param[2]}_Zeta{param[3]}_{param[4]}_{param[5]}_Train.hdf5'
            ),
            initial_step=initial_steps,
            saved_folder=f'/pscratch/sd/m/mansisak/PDEBench/pdebench_data/2D/CFD/2D_Train_{param[0]}/',
            # Rand/',
            train=train,
            transform=data_transforms,
            batch_size=batch_size,
            resolution_proportions=resolution_ratios,
            seed=seed,
        )
    if dataset_name == 'incomp_ns_pdebench':
        dataset = IncompNSPDEBench(
            filename='full_data_merge.h5',
            initial_step=initial_steps,
            saved_folder='/pscratch/sd/m/mansisak/PDEBench/pdebench_data/2D/NS_incom/',
            train=train,
            transform=data_transforms,
            batch_size=batch_size,
            resolution_proportions=resolution_ratios,
            seed=seed,
        )

    return dataset


def get_data(
    **data_args: typing.Any,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Get data w/ args."""
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
    # NOTE(MS): single dataset standard
    # intended use: HP search, filter/downsample exp
    test_kwargs = data_args
    test_kwargs['train'] = False
    if test_kwargs['test_res'] == 'single':
        test_dataset = get_dataset(**test_kwargs)
        test_datasets['test'] = test_dataset

    # multiple test_datasets
    # NOTE(MS): (this may not work w/ multiple downsample/filter regeims)
    # intended us: multi-res training testing
    if test_kwargs['test_res'] == 'multi':
        for res in range(4):
            resolution_ratios = [0, 0, 0, 0]
            resolution_ratios[res] = 1
            test_kwargs['resolution_ratios'] = resolution_ratios
            test_dataset = get_dataset(**test_kwargs)
            test_datasets[f'test_res_{res}'] = test_dataset

    training_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        prefetch_factor=3,
        num_workers=3,
    )

    testing_loaders = {}
    for k, test_dataset in test_datasets.items():
        testing_loaders[k] = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
            prefetch_factor=8,
            num_workers=1,
        )
    return (training_loader, testing_loaders)
