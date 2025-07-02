"""Experiment script: train models."""

from __future__ import annotations

import argparse

import parsl
from parsl.app.app import bash_app
from parsl.app.app import python_app
from parsl_setup import get_parsl_config


@bash_app
def hello_world() -> str:
    """Hello World."""
    exec_str = 'echo HIII'

    return exec_str


@python_app
def check_cuda(random_num: int) -> int:
    """Check if cuda is on device ."""
    import sys

    import torch

    gpu_avail = torch.cuda.is_available()
    print(f'{random_num=}, {gpu_avail=}', file=sys.stderr)

    return 0


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

    experiment_args = []
    for num in range(16):
        experiment_args.append({'random_num': num})

    config = get_parsl_config()

    with parsl.load(config):
        futures = [check_cuda(**exp_args) for exp_args in experiment_args]

        for future in futures:
            print(f'Waiting for {future}')
            print(f'Got result {future.result()}')
