"""Function to get experiment args."""

from __future__ import annotations

import typing


def get_multi_res_args() -> list[dict[str, typing.Any]]:
    """Get Training Params for basic multi res experiment."""
    hyper_param_search_args = []
    for dataset_name in [
        'darcy_pdebench',
        #'burgers_pdebench',
        #'incomp_ns_pdebench',
        #'ns_pdebench',
    ]:
        model_name = 'FNO2D'
        # num time steps * channels:
        in_channels = 10
        out_channels = 1
        initial_steps = 10
        wd = 1e-7

        if dataset_name == 'darcy_pdebench':
            img_size = 128
            batch_size = 128
            in_channels = 1
            initial_steps = 1
            lr = 1e-3

        # Add hyper-parameter search:
        for res_ratio in [
            '[0.1,0.1,0.1,0.7]',
            '[0.1,0,0,0.9]',
            '[0.02,0.03,0.05,0.9]',
        ]:
            hp_args = {
                'lr': lr,
                'weight_decay': wd,
                'step_size': 15,
                'gamma': 0.5,
                'loss_name': 'mse',
                'batch_size': batch_size,
                'dataset_name': dataset_name,
                'downsample_dim': -1,
                'filter_lim': -1,
                'max_mode': img_size // 2,
                'model_name': model_name,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'pinn_loss_weight': 0.5,  # irrelavent arg
                'initial_steps': initial_steps,
                'test_res': 'multi',
                'resolution_ratios': res_ratio,  # high to low
            }
            hyper_param_search_args.append(hp_args)

    return hyper_param_search_args


def get_filter_downsample_args() -> list[dict[str, typing.Any]]:
    """Get Training Params for basic filter/downsample experiment."""
    train_args = []
    for dataset_name in [
        'darcy_pdebench',
        #'burgers_pdebench',
        #'incomp_ns_pdebench',
        #'ns_pdebench',
    ]:
        model_name = 'FNO2D'
        # num time steps * channels:
        in_channels = 10
        out_channels = 1
        initial_steps = 10
        wd = 1e-7

        if dataset_name == 'darcy_pdebench':
            img_size = 128
            batch_size = 128
            in_channels = 1
            initial_steps = 1
            fixed_lim = 8
            filter_lims = [8, 16, 32]  # -1 finished
            downsample_dims = [32, 64, -1]  # 16 finished
            lr = 1e-3

        if dataset_name == 'burgers_pdebench':
            img_size = 1024
            batch_size = 64
            model_name = 'FNO1D'
            fixed_lim = 64
            filter_lims = [64, 128, 256]  # -1 finished
            downsample_dims = [256, 512, -1]  # 128 finished
            lr = 1e-4

        if dataset_name == 'incomp_ns_pdebench':
            # TODO(MS): waiting for HP search
            img_size = 510
            batch_size = 4
            fixed_lim = 85 // 2  # half of 85 // 2
            filter_lims = [85 // 2, 255 // 2]  # -1 finished
            downsample_dims = [255, -1]  # 85 finished
            lr = 1e-4

        if dataset_name == 'ns_pdebench':
            # TODO(MS): waiting for HP search
            img_size = 512
            batch_size = 4
            in_channels = 40
            out_channels = 4
            fixed_lim = 16
            filter_lims = [16, 32, 64]  # -1 finished
            downsample_dims = [128, 256, -1]  # 64 finished
            lr = 1e-4

        # if dataset_name == 'darcy':
        #    fixed_lim = 3
        #    img_size = 32
        #    downsample_dims = [-1, 6, 8, 12]
        #    filter_lims = [-1, 10, 5, 3]

        # study effect of downsampling
        for downsample_dim in downsample_dims:
            training_args = {
                'lr': lr,
                'weight_decay': wd,
                'step_size': 15,
                'gamma': 0.5,
                'loss_name': 'mse',
                'batch_size': batch_size,
                'dataset_name': dataset_name,
                'downsample_dim': downsample_dim,
                'filter_lim': fixed_lim,
                'max_mode': img_size // 2,
                'model_name': model_name,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'pinn_loss_weight': 0.5,  # irrelavent arg
                'initial_steps': initial_steps,
                'test_res': 'multi',
                'resolution_ratios': '[1,0,0,0]',  # high to low
            }
            train_args.append(training_args)
        # study effect of filtering
        for filter_lim in filter_lims:
            training_args = {
                'lr': lr,
                'weight_decay': wd,
                'step_size': 15,
                'gamma': 0.5,
                'loss_name': 'mse',
                'batch_size': batch_size,
                'dataset_name': dataset_name,
                'downsample_dim': -1,
                'filter_lim': filter_lim,
                'max_mode': img_size // 2,
                'model_name': model_name,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'pinn_loss_weight': 0.5,  # irrelavent arg
                'initial_steps': initial_steps,
                'test_res': 'multi',
                'resolution_ratios': '[1,0,0,0]',  # high to low
            }
            train_args.append(training_args)

    return train_args


def get_hp_search_args() -> list[dict[str, typing.Any]]:
    """Get Training Params for PINO w/ HP search."""
    hyper_param_search_args = []
    for dataset_name in [
        #'incomp_ns_pdebench',
        #'ns_pdebench',
        #'darcy_pdebench',
        'burgers_pdebench',
    ]:
        model_name = 'FNO2D'
        in_channels = 10
        out_channels = 1
        initial_steps = 10
        # if dataset_name == 'incomp_ns_pdebench':

        # incomp_ns args
        img_size = 255
        pinn_loss_name = 'incomp_ns_pinn'
        batch_size = 4

        if dataset_name == 'ns_pdebench':
            img_size = 256
            pinn_loss_name = 'n/a'
            batch_size = 4
            # num time steps * channels:
            in_channels = 40
            # num channels:
            out_channels = 4

        if dataset_name == 'darcy_pdebench':
            img_size = 64
            pinn_loss_name = 'darcy_pinn'
            batch_size = 128
            in_channels = 1
            initial_steps = 1

        if dataset_name == 'burgers_pdebench':
            img_size = 512
            pinn_loss_name = 'burgers_pinn'
            batch_size = 64
            model_name = 'FNO1D'

        # Add hyper-parameter search:
        for loss_name in ['mse', pinn_loss_name]:
            # NOTE(MS): we don't do pinns loss for compressible NS
            # we will just let the pinns loss error out for NS
            # if loss_name == 'n/a':
            #    continue
            pinn_loss_weights = [0.5]
            if 'pinn' in loss_name:
                pinn_loss_weights += [0.25, 0.1]  # 0.75 was too high
            for pinn_loss_weight in pinn_loss_weights:
                for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
                    for wd in [1e-5, 1e-6, 1e-7]:
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
                            'max_mode': img_size // 2,
                            'model_name': model_name,
                            'in_channels': in_channels,
                            'out_channels': out_channels,
                            'pinn_loss_weight': pinn_loss_weight,
                            'initial_steps': initial_steps,
                            'test_res': 'single',
                            'resolution_ratios': '[0,1,0,0]',  # high to low
                        }
                        hyper_param_search_args.append(hp_args)
    return hyper_param_search_args
