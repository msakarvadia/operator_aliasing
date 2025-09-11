"""Utility functions for fetching and managing models."""

from __future__ import annotations

import typing

from neuralop.models import FNO
from torch.nn import Module

from operator_aliasing.models.cno2d import CNO2d
from operator_aliasing.models.crop2d import CROPFNO2d


def get_model(**model_args: typing.Any) -> Module:
    """Get model w/ args."""
    model = None
    model_name = model_args['model_name']
    max_modes = model_args['max_modes']
    hidden_channels = model_args['hidden_channels']
    in_channels = model_args['in_channels']
    out_channels = model_args['out_channels']

    # crop specific params
    # in_size = model_args['img_size']
    latent_size = model_args['latent_size']

    if model_name == 'CNO2D':
        # NOTE (MS): default recommended from
        # original repo
        # https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/main/CNO2d_vanilla_torch_version/CNO2d.py#L203 # noqa
        model = CNO2d(
            in_dim=in_channels,
            out_dim=out_channels,
            n_layers=1,
            n_res=4,
            n_res_neck=4,
            channel_multiplier=hidden_channels,
            use_bn=True,
        )
    if model_name == 'CROP2D':
        starting_modes: typing.Any = (max_modes, max_modes)
        model = CROPFNO2d(
            modes=starting_modes,
            width=hidden_channels,
            # in_size=in_size,
            latent_size=latent_size,
            time_steps=in_channels,
        )
    if model_name == 'FNO2D':
        starting_modes = (max_modes, max_modes)
        model = FNO(
            n_modes=starting_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    if model_name == 'FNO1D':
        starting_modes = (max_modes,)
        model = FNO(
            n_modes=starting_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    return model
