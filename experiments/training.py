"""Experiment script: train models."""

from __future__ import annotations

import parsl
from parsl.app.app import bash_app
from parsl_setup import get_parsl_config


@bash_app
def hello_world() -> str:
    """Hello World."""
    exec_str = 'echo HIII'

    return exec_str


@bash_app
def train(filter_lim: int, downsample_dim: int) -> str:
    """Train a model."""
    exec_str = f"""pwd;
    python main.py --filter_lim {filter_lim} \n
    --downsample_dim {downsample_dim} \n
    --ckpt_path ckpts/{downsample_dim}_{filter_lim}/
    """
    return exec_str


if __name__ == '__main__':
    config = get_parsl_config()
    parsl.load(config)

    train_args = []
    for downsample_dim in [-1, 8, 11]:
        for filter_lim in [-1, 5]:
            # don't downsample unfiltered data
            if filter_lim == -1 and downsample_dim != -1:
                continue
            train_args.append(
                {'downsample_dim': downsample_dim, 'filter_lim': filter_lim}
            )

    futures = [train(**args) for args in train_args]
    for future in futures:
        print(f'Waiting for {future}')
        print(f'Got result {future.result()}')
