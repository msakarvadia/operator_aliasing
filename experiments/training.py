"""Experiment script: train models."""

from __future__ import annotations

import argparse
import typing

import parsl
from get_train_args import get_train_args
from parsl.app.app import bash_app
from parsl_setup import get_parsl_config


@bash_app
def hello_world() -> str:
    """Hello World."""
    exec_str = 'echo HIII'

    return exec_str


@bash_app
def train(**kwargs: typing.Any) -> str:
    """Train a model."""
    filter_lim = kwargs['filter_lim']
    downsample_dim = kwargs['downsample_dim']
    dataset_name = kwargs['dataset_name']
    lr = kwargs['lr']
    wd = kwargs['weight_decay']
    step_size = kwargs['step_size']
    gamma = kwargs['gamma']
    loss_name = kwargs['loss_name']
    max_mode = kwargs['max_mode']
    batch_size = kwargs['batch_size']

    arg_path = '_'.join(map(str, list(kwargs.values())))
    # Need to remove any . or / to
    # ensure a single continuous file path
    arg_path = arg_path.replace('.', '')
    ckpt_name = arg_path.replace('/', '')

    if dataset_name == 'darcy':
        exec_str = f"""pwd;
        python main.py --filter_lim {filter_lim} \
        --downsample_dim {downsample_dim} \
        --lr {lr} \
        --weight_decay {wd} \
        --step_size {step_size} \
        --gamma {gamma} \
        --ckpt_path ckpts/{ckpt_name} \
        --loss_name {loss_name} \
        --max_modes {max_mode} \
        --batch_size {batch_size} \
        """
    if dataset_name == 'darcy_pdebench':
        exec_str = f"""pwd;
        python main.py --filter_lim {filter_lim} \
        --downsample_dim {downsample_dim} \
        --lr {lr} \
        --weight_decay {wd} \
        --step_size {step_size} \
        --gamma {gamma} \
        --dataset_name darcy_pdebench \
        --img_size 128 \
        --ckpt_path ckpts/{ckpt_name} \
        --loss_name {loss_name} \
        --loss_name {loss_name} \
        --max_modes {max_mode} \
        """
    return exec_str


if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='darcy',
        choices=['darcy', 'darcy_pdebench'],
        help='Name of training data.',
    )
    parser.add_argument(
        '--queue',
        type=str,
        default='debug',
        choices=['debug', 'regular'],
        help='Name of slurm queue we want to run in.',
    )
    parser.add_argument(
        '--walltime',
        type=str,
        default='00:30:00',
        help='HH:MM:SS length of job. Check you cluster queue limits.',
    )
    parser.add_argument(
        '--num_nodes',
        type=int,
        default=1,
        help='Number of nodes in your job.',
    )
    args = parser.parse_args()

    """
    train_args = []
    if args.dataset_name == 'darcy':
        fixed_lim = 3
        downsample_dims = [-1, 6, 8, 12]
        max_modes = [4, 8, 16]
        filter_lims = [-1, 10, 5, 3]
    if args.dataset_name == 'darcy_pdebench':
        fixed_lim = 8
        downsample_dims = [-1, 16, 32, 64]
        max_modes = [16, 32, 64]
        filter_lims = [-1, 8, 16, 32]

    # study effect of downsampling
    for downsample_dim in downsample_dims:
        training_args = {
            'dataset_name': args.dataset_name,
            'downsample_dim': downsample_dim,
            'filter_lim': fixed_lim,
        }
        train_args.append(training_args)
    # study effect of filtering
    for filter_lim in filter_lims:
        training_args = {
            'dataset_name': args.dataset_name,
            'downsample_dim': -1,
            'filter_lim': filter_lim,
        }
        train_args.append(training_args)

    # Add hyper-parameter search:
    hyper_param_search_args = []
    for train_arg in train_args:
        for loss_name in ['l1']:
            for max_mode in max_modes:
                for lr in [1e-3]:  # 1e-2, 1e-3, 1e-5
                    for wd in [1e-8]:  # 1e-7, 1e-8, 1e-9
                        for step_size in [15]:
                            for gamma in [0.5]:
                                experiment_args = train_arg.copy()
                                hp_args = {
                                    'lr': lr,
                                    'weight_decay': wd,
                                    'step_size': step_size,
                                    'gamma': gamma,
                                    'loss_name': loss_name,
                                    'max_mode': max_mode,
                                    'batch_size': 32,
                                }
                                hyper_param_search_args.append(
                                    experiment_args | hp_args
                                )
    """
    training_args = get_train_args(args.dataset_name)

    config = get_parsl_config(
        walltime=args.walltime, queue=args.queue, num_nodes=args.num_nodes
    )
    with parsl.load(config):
        futures = [train(**args) for args in training_args]
        print(f'Num of experiments: {len(futures)}')

        # futures = futures + pinn_futures
        for future in futures:
            print(f'Waiting for {future}')
            print(f'Got result {future.result()}')
