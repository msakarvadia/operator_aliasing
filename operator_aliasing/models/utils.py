"""Utility functions for fetching and managing models."""

from __future__ import annotations

import typing

from torch.nn import Module


def get_model(model_name: str, **model_args: typing.Any) -> Module:
    """Get model w/ args."""
    return None
