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


if __name__ == '__main__':
    config = get_parsl_config()
    parsl.load(config)

    future = hello_world()
    print(f'Waiting for {future}')
    print(f'Got result {future.result()}')
