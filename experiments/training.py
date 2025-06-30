"""Experiment script: train models."""

from __future__ import annotations

import argparse

import parsl
from parsl.app.app import bash_app
from parsl_setup import get_parsl_config


@bash_app
def hello_world() -> str:
    """Hello World."""
    exec_str = 'echo HIII'

    return exec_str


@bash_app
def train(dataset_name: str, filter_lim: int, downsample_dim: int) -> str:
    """Train a model."""
    if dataset_name == 'darcy':
        exec_str = f"""pwd;
        python main.py --filter_lim {filter_lim} \
        --downsample_dim {downsample_dim} \
        --ckpt_path ckpts/{downsample_dim}_{filter_lim} \
        """
    if dataset_name == 'darcy_pdebench':
        exec_str = f"""pwd;
        python main.py --filter_lim {filter_lim} \
        --downsample_dim {downsample_dim} \
        --ckpt_path ckpts/{downsample_dim}_{filter_lim} \
        --dataset_name darcy_pdebench \
        --img_size 128 \
        --ckpt_path darcy_pdebench_ckpts/{downsample_dim}_{filter_lim} \
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
    args = parser.parse_args()

    config = get_parsl_config()
    parsl.load(config)

    train_args = []
    for downsample_dim in [-1, 6, 8, 12]:
        for filter_lim in [-1, 10, 5, 4, 3]:
            # don't downsample unfiltered data
            if filter_lim == -1 and downsample_dim != -1:
                continue
            train_args.append(
                {
                    'dataset_name': args.dataset_name,
                    'downsample_dim': downsample_dim,
                    'filter_lim': filter_lim,
                }
            )

    futures = [train(**args) for args in train_args]
    for future in futures:
        print(f'Waiting for {future}')
        print(f'Got result {future.result()}')
