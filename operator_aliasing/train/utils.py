"""Training utility functions."""

from __future__ import annotations

import glob
import os
import typing

import pandas as pd
import torch
from torch import nn

from .pinn_losses import BurgersDataAndPinnsLoss
from .pinn_losses import DarcyDataAndPinnsLoss
from .pinn_losses import IncompNSDataAndPinnsLoss
from .pinn_losses import Loss


def get_loss(
    loss_name: str,
    pinn_loss_weight: float,
    darcy_forcing_term: float,
    burger_viscosity: float,
    incomp_ns_viscosity: float,
) -> nn.Module:
    """Get loss functions."""
    loss = None
    if loss_name in ['mse', 'l1']:
        loss = Loss(loss_name)
    if loss_name == 'darcy_pinn':
        loss = DarcyDataAndPinnsLoss(pinn_loss_weight, darcy_forcing_term)
    if loss_name == 'burgers_pinn':
        loss = BurgersDataAndPinnsLoss(pinn_loss_weight, burger_viscosity)
    if loss_name == 'incomp_ns_pinn':
        loss = IncompNSDataAndPinnsLoss(pinn_loss_weight, incomp_ns_viscosity)
    return loss


def save_ckpt(ckpt_path: str, ckpt_dict: typing.Any) -> None:
    """Ckpt model during training."""
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # save train stats as csv, not in PT ckpt obj
    train_stats = ckpt_dict['train_stats']
    train_stats.to_csv(f'{ckpt_path}/train_stats.csv', index=False)
    ckpt_dict.pop('train_stats')

    ckpt_num = ckpt_dict['epoch']
    torch.save(ckpt_dict, f'{ckpt_path}/{ckpt_num}_ckpt.pth')


def load_latest_ckpt(ckpt_path: str) -> typing.Any:
    """Load ckpt if it exists."""
    list_of_ckpts = glob.glob(f'{ckpt_path}/*.pth')
    if list_of_ckpts:
        latest_ckpt = max(list_of_ckpts, key=os.path.getctime)
        print(f'Resuming training from {latest_ckpt}')
        ckpt_dict = torch.load(
            latest_ckpt,
            weights_only=False,
        )

        # load and return train stats as dataframe
        train_stats = pd.read_csv(f'{ckpt_path}/train_stats.csv')
        ckpt_dict['train_stats'] = train_stats
        return ckpt_dict
    else:
        return None
