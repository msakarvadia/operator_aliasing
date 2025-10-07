"""Experiment script: train models."""

from __future__ import annotations

import argparse
import typing

import parsl
from get_train_args import get_filter_downsample_args
from get_train_args import get_hp_search_alias_free
from get_train_args import get_hp_search_args
from get_train_args import get_multi_res_args
from get_train_args import get_pino_args
from get_train_args import get_timing_multi_res_args
from get_train_args import get_train_alias_free
from parsl.app.app import bash_app
from parsl_setup import get_parsl_config


@bash_app
def hello_world() -> str:
    """Hello World."""
    exec_str = 'echo HIII'

    return exec_str


@bash_app
def train(ckpt_dir: str = 'ckpts', **kwargs: typing.Any) -> str:
    """Train a model."""
    epochs = 150
    if 'epochs' in kwargs:
        epochs = kwargs['epochs']
        kwargs.pop('epochs')
    arg_path = '_'.join(map(str, list(kwargs.values())))
    # Need to remove any . or / to
    # ensure a single continuous file path
    arg_path = arg_path.replace('.', '')
    arg_path = arg_path.replace('[', '')
    arg_path = arg_path.replace(']', '')
    arg_path = arg_path.replace(',', '_')
    ckpt_name = arg_path.replace('/', '')

    exec_str = f"""pwd;
    python main.py --filter_lim {kwargs['filter_lim']} \
    --downsample_dim {kwargs['downsample_dim']} \
    --lr {kwargs['lr']} \
    --weight_decay {kwargs['weight_decay']} \
    --step_size {kwargs['step_size']} \
    --gamma {kwargs['gamma']} \
    --dataset_name {kwargs['dataset_name']} \
    --ckpt_path {ckpt_dir}/{ckpt_name} \
    --loss_name {kwargs['loss_name']} \
    --max_modes {kwargs['max_mode']} \
    --batch_size {kwargs['batch_size']} \
    --model_name {kwargs['model_name']}\
    --out_channels {kwargs['out_channels']} \
    --in_channels {kwargs['in_channels']} \
    --initial_steps {kwargs['initial_steps']} \
    --pinn_loss_weight {kwargs['pinn_loss_weight']} \
    --test_res {kwargs['test_res']} \
    --resolution_ratios {kwargs['resolution_ratios']}\
    --epochs {epochs}\
    """

    if 'latent_size' in kwargs:
        exec_str = f"""
        python main.py --filter_lim {kwargs['filter_lim']} \
        --downsample_dim {kwargs['downsample_dim']} \
        --lr {kwargs['lr']} \
        --weight_decay {kwargs['weight_decay']} \
        --step_size {kwargs['step_size']} \
        --gamma {kwargs['gamma']} \
        --dataset_name {kwargs['dataset_name']} \
        --ckpt_path {ckpt_dir}/{ckpt_name} \
        --loss_name {kwargs['loss_name']} \
        --max_modes {kwargs['max_mode']} \
        --batch_size {kwargs['batch_size']} \
        --model_name {kwargs['model_name']}\
        --out_channels {kwargs['out_channels']} \
        --in_channels {kwargs['in_channels']} \
        --initial_steps {kwargs['initial_steps']} \
        --pinn_loss_weight {kwargs['pinn_loss_weight']} \
        --test_res {kwargs['test_res']} \
        --resolution_ratios {kwargs['resolution_ratios']}\
        --latent_size {kwargs['latent_size']}\
        --img_size {kwargs['img_size']}\
        --epochs {epochs}\
        """

    return exec_str


if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='filter_downsample',
        choices=[
            'filter_downsample',
            'hp_search',
            'pino',
            'multi_res',
            'alias_free',
            'alias_free_train',
            'timing',
        ],
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
        '--ckpt_dir',
        type=str,
        default='timing_ckpts',
        choices=['timing_ckpts', 'ns_ckpts', 'ckpts'],
        help='Name of dir to store all experiments in.',
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
    parser.add_argument(
        '--gpu_mem',
        type=int,
        default=40,
        choices=[40, 80],
        help='Number of GB mem on GPU being requested.',
    )
    args = parser.parse_args()

    if args.experiment_name == 'filter_downsample':
        training_args = get_filter_downsample_args()
    if args.experiment_name == 'hp_search':
        training_args = get_hp_search_args()
    if args.experiment_name == 'multi_res':
        training_args = get_multi_res_args()
    if args.experiment_name == 'timing':
        training_args = get_timing_multi_res_args()
    if args.experiment_name == 'pino':
        training_args = get_pino_args()
    if args.experiment_name == 'alias_free':
        training_args = get_hp_search_alias_free()
    if args.experiment_name == 'alias_free_train':
        training_args = get_train_alias_free()

    config = get_parsl_config(
        walltime=args.walltime,
        queue=args.queue,
        num_nodes=args.num_nodes,
        gpu_mem=args.gpu_mem,
    )
    with parsl.load(config):
        futures = [
            train(ckpt_dir=args.ckpt_dir, **exp_args)
            for exp_args in training_args
        ]
        print(f'Num of experiments: {len(futures)}')

        for train_args, future in zip(training_args, futures):
            print(train_args)
            print(f'Waiting for {future}')
            print(f'Got result {future.result()}')
