"""Training utility functions."""

from __future__ import annotations

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


# def load_latest_ckpt(ckpt_path:str) -> :
