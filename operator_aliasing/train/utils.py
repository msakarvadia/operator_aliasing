"""Training utility functions."""

from __future__ import annotations

from torch import nn


def get_loss(loss_name: str) -> nn.Module:
    """Get loss functions."""
    loss = None
    if loss_name == 'l1':
        loss = nn.L1Loss()
    return loss
