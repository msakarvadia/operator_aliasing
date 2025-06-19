"""foo module."""

from __future__ import annotations

import argparse

import torch
from neuralop.models import FNO
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

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

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Get Model
    max_modes = 16
    starting_modes = (max_modes, max_modes)
    model = FNO(
        max_n_modes=(max_modes, max_modes),
        n_modes=starting_modes,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
    ).to(device)
    args.model = model

    # Get DataLoaders
    n_train = 100  # number of training samples

    x_data = torch.rand((256, 1, 128, 128)).type(torch.float32)
    y_data = torch.ones((256, 1, 128, 128)).type(torch.float32)

    input_function_train = x_data[:n_train, :]
    output_function_train = y_data[:n_train, :]
    input_function_test = x_data[n_train:, :]
    output_function_test = y_data[n_train:, :]

    batch_size = args.batch_size

    training_set = DataLoader(
        TensorDataset(input_function_train, output_function_train),
        batch_size=batch_size,
        shuffle=True,
    )
    testing_set = DataLoader(
        TensorDataset(input_function_test, output_function_test),
        batch_size=batch_size,
        shuffle=False,
    )
    args.train_dataloader = training_set
    args.test_dataloader = testing_set

    # Get Loss Function
    args.loss = get_loss(args.loss_name)

    # Train Model
    train_model(**vars(args))
