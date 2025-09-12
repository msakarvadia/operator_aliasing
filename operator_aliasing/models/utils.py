"""Utility functions for fetching and managing models."""

from __future__ import annotations

import typing

from neuralop.models import FNO
from torch.nn import Module

from operator_aliasing.models.CNO2d_original_version.CNOModule import CNO
from operator_aliasing.models.crop2d import CROPFNO2d


def get_model(**model_args: typing.Any) -> Module:
    """Get model w/ args."""
    model = None
    model_name = model_args['model_name']
    max_modes = model_args['max_modes']
    hidden_channels = model_args['hidden_channels']
    in_channels = model_args['in_channels']
    out_channels = model_args['out_channels']

    # crop + CNO specific params
    latent_size = model_args['latent_size']

    # CNO params
    in_size = model_args['img_size']

    if model_name == 'CNO2D':
        # defaults from
        # https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/main/CNO2d_original_version/CNOModule.py#L261 # noqa
        model = CNO(
            in_dim=in_channels,  # Number of input channels.
            in_size=in_size,  # Input spatial size
            N_layers=1,  # Number of (D) or (U) blocks in the network
            N_res=1,  # Number of (R) blocks per level (except the neck)
            N_res_neck=6,  # Number of (R) blocks in the neck
            channel_multiplier=32,  # How the number of channels evolve?
            conv_kernel=3,  # Size of all the kernels
            cutoff_den=2.0001,  # Filter property 1.
            filter_size=6,  # Filter property 2.
            lrelu_upsampling=2,  # Filter property 3.
            half_width_mult=0.8,  # Filter property 4.
            radial=False,  # Filter property 5. Is filter radial?
            batch_norm=True,  # Add BN? We do not add BN in lifting/projection layer # noqa
            out_dim=out_channels,  # Target dimension
            out_size=1,  # If out_size is 1, Then out_size = in_size. Else must be int # noqa
            expand_input=False,  # Start with original in_size, or expand it (pad zeros in the spectrum) # noqa
            latent_lift_proj_dim=latent_size,  # Intermediate latent dimension in the lifting/projection layer # noqa
            add_inv=True,  # Add invariant block (I) after the intermediate connections? # noqa
            activation='cno_lrelu_torch',  # Activation function can be 'cno_lrelu' or 'lrelu' # noqa
            # activation='lrelu'
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
