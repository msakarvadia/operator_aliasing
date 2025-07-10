"""Function to get experiment args."""

from __future__ import annotations

import typing


def get_filter_downsample_args(
    dataset_name: str,
) -> list[dict[str, typing.Any]]:
    """Get Training Params for basic filter/downsample experiment."""
    train_args = []
    if dataset_name == 'darcy':
        fixed_lim = 3
        img_size = 32
        downsample_dims = [-1, 6, 8, 12]
        max_modes = [4, 8, 16]
        filter_lims = [-1, 10, 5, 3]
    if dataset_name == 'darcy_pdebench':
        fixed_lim = 8
        img_size = 128
        downsample_dims = [16, 32, 64, -1]
        max_modes = [16, 32, 64]
        filter_lims = [8, 16, 32, -1]

    # study effect of downsampling
    for downsample_dim in downsample_dims:
        training_args = {
            'dataset_name': dataset_name,
            'downsample_dim': downsample_dim,
            'filter_lim': fixed_lim,
            'img_size': img_size,
        }
        train_args.append(training_args)
    # study effect of filtering
    for filter_lim in filter_lims:
        training_args = {
            'dataset_name': dataset_name,
            'downsample_dim': -1,
            'filter_lim': filter_lim,
            'img_size': img_size,
        }
        train_args.append(training_args)

    # Add hyper-parameter search:
    hyper_param_search_args = []
    for train_arg in train_args:
        for loss_name in ['l1']:
            for max_mode in max_modes:
                for lr in [1e-3]:  # 1e-2, 1e-3, 1e-5
                    for wd in [1e-8]:  # 1e-7, 1e-8, 1e-9
                        experiment_args = train_arg.copy()
                        hp_args = {
                            'lr': lr,
                            'weight_decay': wd,
                            'step_size': 15,
                            'gamma': 0.5,
                            'loss_name': loss_name,
                            'max_mode': max_mode,
                            'batch_size': 32,
                        }
                        hyper_param_search_args.append(
                            experiment_args | hp_args
                        )
    return hyper_param_search_args


def get_pino_args(dataset_name: str) -> list[dict[str, typing.Any]]:
    """Get Training Params for PINO w/ HP search."""
    assert dataset_name == 'darcy', 'No support for darcy_pdebench right now.'

    if dataset_name == 'darcy':
        img_sizes = [16, 32, 64, 128]

    # Add hyper-parameter search:
    hyper_param_search_args = []
    for img_size in img_sizes:
        for loss_name in ['l1', 'darcy_pinn']:
            for lr in [1e-2, 1e-3, 1e-5]:
                for wd in [1e-7, 1e-8, 1e-9]:
                    hp_args = {
                        'lr': lr,
                        'weight_decay': wd,
                        'step_size': 15,
                        'gamma': 0.5,
                        'loss_name': loss_name,
                        'batch_size': 32,
                        'dataset_name': dataset_name,
                        'downsample_dim': -1,
                        'filter_lim': -1,
                        'img_size': img_size,
                        'max_mode': img_size // 2,
                    }
                    hyper_param_search_args.append(hp_args)
    return hyper_param_search_args
