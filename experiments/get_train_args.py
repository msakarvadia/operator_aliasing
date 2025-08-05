"""Function to get experiment args."""

from __future__ import annotations

import typing


def get_filter_downsample_args() -> list[dict[str, typing.Any]]:
    """Get Training Params for basic filter/downsample experiment."""
    hyper_param_search_args = []
    for dataset_name in [
        'darcy_pdebench',
        'burgers_pdebench',
        'incomp_ns_pdebench',
    ]:
        train_args = []
        model_name = 'FNO2D'
        # num time steps * channels:
        in_channels = 10
        initial_steps = 10

        if dataset_name == 'darcy_pdebench':
            img_size = 128
            batch_size = 128
            in_channels = 1
            initial_steps = 1
            fixed_lim = 8
            filter_lims = [8, 16, 32]  # -1 finished
            downsample_dims = [32, 64, -1]  # 16 finished
            lr = 1e-3
            wd = 1e-7

        if dataset_name == 'burgers_pdebench':
            img_size = 1024
            batch_size = 64
            model_name = 'FNO1D'
            fixed_lim = 64
            filter_lims = [64, 128, 256]  # -1 finished
            downsample_dims = [256, 512, -1]  # 16 finished
            lr = 1e-4
            wd = 1e-7

        if dataset_name == 'incomp_ns_pdebench':
            # TODO(MS): waiting for HP search
            img_size = 510
            batch_size = 4
            fixed_lim = 85 // 2  # half of 85 // 2
            filter_lims = [85 // 2, 255 // 2]  # -1 finished
            downsample_dims = [255, -1]  # 85 finished
            lr = 1e-4
            wd = 1e-7

        if dataset_name == 'darcy':
            fixed_lim = 3
            img_size = 32
            downsample_dims = [-1, 6, 8, 12]
            filter_lims = [-1, 10, 5, 3]

        # study effect of downsampling
        for downsample_dim in downsample_dims:
            training_args = {
                'dataset_name': dataset_name,
                'downsample_dim': downsample_dim,
                'filter_lim': fixed_lim,
                'img_size': img_size,
                'max_mode': img_size // 2,
            }
            train_args.append(training_args)
        # study effect of filtering
        for filter_lim in filter_lims:
            training_args = {
                'dataset_name': dataset_name,
                'downsample_dim': -1,
                'filter_lim': filter_lim,
                'img_size': img_size,
                'max_mode': img_size // 2,
            }
            train_args.append(training_args)

        # Add hyper-parameter search:
        for train_arg in train_args:
            experiment_args = train_arg.copy()
            hp_args = {
                'lr': lr,
                'weight_decay': wd,
                'step_size': 15,
                'gamma': 0.5,
                'loss_name': 'l1',
                'batch_size': batch_size,
                'model_name': model_name,
                'in_channels': in_channels,
                'initial_steps': initial_steps,
            }
            hyper_param_search_args.append(experiment_args | hp_args)

    return hyper_param_search_args


def get_pino_args() -> list[dict[str, typing.Any]]:
    """Get Training Params for PINO w/ HP search."""
    hyper_param_search_args = []
    for dataset_name in [
        'darcy_pdebench',
        'burgers_pdebench',
        'incomp_ns_pdebench',
    ]:
        model_name = 'FNO2D'
        # num time steps * channels:
        in_channels = 10
        initial_steps = 10

        if dataset_name == 'darcy_pdebench':
            img_sizes = [16, 32, 64, 128]
            pinn_loss_name = 'darcy_pinn'
            batch_size = 128
            in_channels = 1
            initial_steps = 1

        if dataset_name == 'burgers_pdebench':
            img_sizes = [128, 256, 512, 1024]
            pinn_loss_name = 'burgers_pinn'
            batch_size = 64
            model_name = 'FNO1D'

        if dataset_name == 'incomp_ns_pdebench':
            img_sizes = [17, 85, 255, 510]
            pinn_loss_name = 'incomp_ns_pinn'
            batch_size = 4

        # Add hyper-parameter search:
        for img_size in img_sizes:
            for loss_name in ['l1', pinn_loss_name]:
                for lr in [1e-3, 1e-4, 1e-5]:
                    for wd in [1e-7, 1e-8, 1e-9]:
                        hp_args = {
                            'lr': lr,
                            'weight_decay': wd,
                            'step_size': 15,
                            'gamma': 0.5,
                            'loss_name': loss_name,
                            'batch_size': batch_size,
                            'dataset_name': dataset_name,
                            'downsample_dim': -1,
                            'filter_lim': -1,
                            'img_size': img_size,
                            'max_mode': img_size // 2,
                            'model_name': model_name,
                            'in_channels': in_channels,
                            'initial_steps': initial_steps,
                        }
                        hyper_param_search_args.append(hp_args)
    return hyper_param_search_args
