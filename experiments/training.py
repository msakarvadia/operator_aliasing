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
def train() -> str:
    """Train a model."""
    exec_str = 'pwd; python main.py'
    return exec_str


if __name__ == '__main__':
    config = get_parsl_config()
    parsl.load(config)

    future = train()
    print(f'Waiting for {future}')
    print(f'Got result {future.result()}')
