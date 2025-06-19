"""foo module."""

from __future__ import annotations

import argparse

import torch

from operator_aliasing.data.utils import get_data
from operator_aliasing.models.utils import get_model
from operator_aliasing.train.train import train_model
from operator_aliasing.train.utils import get_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Train args
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning Rate for training.',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-8,
        help='Weight decay param for optimizer.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training Epochs.',
    )
    parser.add_argument(
        '--step_size',
        type=int,
        default=15,
        help='LR Scheduler step size.',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.5,
        help='LR Scheduler decay rate.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducability.',
    )
    parser.add_argument(
        '--loss_name',
        type=str,
        default='l1',
        choices=['l1'],
        help='Name of loss functions for training.',
    )

    # Data args
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch Size.',
    )

    # Model args
    parser.add_argument(
        '--max_modes',
        type=int,
        default=16,
        help='FNO: max modes in spectral conv layer.',
    )
    parser.add_argument(
        '--hidden_channels',
        type=int,
        default=32,
        help='FNO: # of hidden channels.',
    )
    parser.add_argument(
        '--in_channels',
        type=int,
        default=1,
        help='FNO: input channel dim.',
    )
    parser.add_argument(
        '--out_channels',
        type=int,
        default=1,
        help='FNO: output channel dim.',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='FNO2D',
        choices=['FNO2D'],
        help='Type of model.',
    )

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Get Model
    args.model = get_model(**vars(args))

    # Get DataLoaders
    (args.train_dataloader, args.test_dataloader) = get_data(**vars(args))

    # Get Loss Function
    args.loss = get_loss(args.loss_name)

    # Train Model
    train_model(**vars(args))
