"""Function to get experiment args."""

from __future__ import annotations

import typing

from operator_aliasing.utils import get_dataset_info


def get_multi_res_args() -> list[dict[str, typing.Any]]:
    """Get Training Params for basic multi res experiment."""
    hyper_param_search_args = []
    for dataset_name in [
        'darcy_pdebench',
        #'burgers_pdebench',
        #'incomp_ns_pdebench',
        #'ns_pdebench',
    ]:
        (
            model_name,
            in_channels,
            out_channels,
            initial_steps,
            loss_name,
            batch_size,
            lr,
            wd,
            _,
        ) = get_dataset_info(dataset_name, 'mse')

        img_size = 510

        if dataset_name == 'ns_pdebench':
            img_size = 512

        if dataset_name == 'darcy_pdebench':
            img_size = 128

        if dataset_name == 'burgers_pdebench':
            img_size = 1024

        # make res ratios
        res_ratios = [
            '[0.1,0.1,0.1,0.7]',
            '[0.1,0,0,0.9]',
            '[0.02,0.03,0.05,0.9]',
        ]
        for rat in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.05]:
            for first_idx, second_idx in [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
            ]:
                ratio: list[float] = [0, 0, 0, 0]
                ratio[first_idx] = rat
                ratio[second_idx] = round(1 - rat, 2)
                ratio_formatted = str(ratio).replace(' ', '')
                res_ratios.append(ratio_formatted)

        # Add hyper-parameter search:
        for res_ratio in res_ratios:
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
        #'incomp_ns_pdebench',
        #'ns_pdebench',
        'burgers_pdebench',
        'darcy_pdebench',
    ]:
        (
            model_name,
            in_channels,
            out_channels,
            initial_steps,
            loss_name,
            batch_size,
            lr,
            wd,
            _,
        ) = get_dataset_info(dataset_name, 'mse')

        if dataset_name == 'darcy_pdebench':
            img_size = 128
            fixed_lim = 8
            filter_lims = [8, 16, 32, -1]  # -1 finished
            downsample_dims = [16, 32, 64, -1]  # 16 finished

        if dataset_name == 'burgers_pdebench':
            img_size = 1024
            model_name = 'FNO1D'
            fixed_lim = 64
            filter_lims = [64, 128, 256, -1]  # -1 finished
            downsample_dims = [128, 256, 512, -1]  # 128 finished

        if dataset_name == 'incomp_ns_pdebench':
            # TODO(MS): waiting for HP search
            img_size = 510
            fixed_lim = 85 // 2  # half of 85 // 2
            filter_lims = [85 // 2, 255 // 2, -1]  # -1 finished
            downsample_dims = [85, 255, -1]  # 85 finished

        if dataset_name == 'ns_pdebench':
            # TODO(MS): waiting for HP search
            img_size = 512
            fixed_lim = 32
            filter_lims = [32, 64, 128, -1]  # -1 finished
            downsample_dims = [64, 128, 256, -1]  # 64 finished

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
                'test_res': 'single',
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
                'test_res': 'single',
                'resolution_ratios': '[1,0,0,0]',  # high to low
            }
            train_args.append(training_args)

    return train_args


def get_pino_args() -> list[dict[str, typing.Any]]:
    """Get Training Params for PINO w/ HP search."""
    hyper_param_search_args = []
    for dataset_name in [
        'darcy_pdebench',
        'incomp_ns_pdebench',
        'ns_pdebench',
        'burgers_pdebench',
    ]:
        # if dataset_name == 'incomp_ns_pdebench':
        img_sizes = [510, 255, 85, 17]

        if dataset_name == 'ns_pdebench':
            img_sizes = [512, 256, 128, 64]

        if dataset_name == 'darcy_pdebench':
            img_sizes = [128, 64, 32, 16]

        if dataset_name == 'burgers_pdebench':
            img_sizes = [1024, 512, 256, 128]

        # Add hyper-parameter search:
        for l_name in ['mse', 'pinn']:
            (
                model_name,
                in_channels,
                out_channels,
                initial_steps,
                loss_name,
                batch_size,
                lr,
                wd,
                pinn_loss_w,
            ) = get_dataset_info(dataset_name, l_name)
            # NOTE(MS): we don't do pinns loss for compressible NS
            # we will just let the pinns loss error out for NS
            # if loss_name == 'n/a':
            #    continue
            pinn_loss_weights = [0.5]
            if 'pinn' in loss_name:
                pinn_loss_weights = [pinn_loss_w]  # 0.75 was too high
                # pinn_loss_weights += [0.25, 0.1]  # 0.75 was too high
            for pinn_loss_weight in pinn_loss_weights:
                for res_idx, resolution_ratios in enumerate(
                    [
                        '[1,0,0,0]',
                        '[0,1,0,0]',
                        '[0,0,1,0]',
                        '[0,0,0,1]',
                    ]
                ):
                    # NOTE(MS): delete later

                    if resolution_ratios == '[0,1,0,0]':
                        # to avoid interference w/ HP search experiment
                        continue
                    img_size = img_sizes[res_idx]
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
                        'resolution_ratios': resolution_ratios,  # high to low
                    }
                    hyper_param_search_args.append(hp_args)
    return hyper_param_search_args


def get_hp_search_args() -> list[dict[str, typing.Any]]:
    """Get Training Params for PINO w/ HP search."""
    hyper_param_search_args = []
    for dataset_name in [
        'incomp_ns_pdebench',
        'ns_pdebench',
        'darcy_pdebench',
        'burgers_pdebench',
    ]:
        # if dataset_name == 'incomp_ns_pdebench':
        img_size = 255

        if dataset_name == 'ns_pdebench':
            img_size = 256

        if dataset_name == 'darcy_pdebench':
            img_size = 64

        if dataset_name == 'burgers_pdebench':
            img_size = 512

        # Add hyper-parameter search:
        for l_name in ['mse', 'pinn']:
            (
                model_name,
                in_channels,
                out_channels,
                initial_steps,
                loss_name,
                batch_size,
                _,
                _,
                _,
            ) = get_dataset_info(dataset_name, l_name)
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
