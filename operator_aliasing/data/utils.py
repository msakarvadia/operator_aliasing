"""Utility functions for fetching and managing data."""

from __future__ import annotations

import typing

from torch.utils.data import DataLoader


def get_data(
    data_name: str, **data_args: typing.Any
) -> tuple[DataLoader, DataLoader]:
    """Get data w/ args."""
    return (None, None)
