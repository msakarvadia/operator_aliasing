"""Training utility functions."""

from __future__ import annotations

import glob
import os
import typing

import torch
from torch import nn


def get_loss(loss_name: str) -> nn.Module:
    """Get loss functions."""
    loss = None
    if loss_name == 'l1':
        loss = nn.L1Loss()
    return loss


def save_ckpt(ckpt_path: str, ckpt_dict: typing.Any) -> None:
    """Ckpt model during training."""
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
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
        return ckpt_dict
    else:
        return None
