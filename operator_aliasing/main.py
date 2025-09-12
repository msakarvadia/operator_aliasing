"""foo module."""

from __future__ import annotations

import argparse

import torch

from operator_aliasing.data.utils import get_data
from operator_aliasing.models.utils import get_model
from operator_aliasing.train.train import train_model
from operator_aliasing.train.utils import get_loss
from operator_aliasing.utils import seed_everything

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
        default=150,
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
        default='mse',
        choices=['mse', 'l1', 'darcy_pinn', 'burgers_pinn', 'incomp_ns_pinn'],
        help='Name of loss functions for training.',
    )
    parser.add_argument(
        '--pinn_loss_weight',
        type=float,
        default=0.5,
        help="""Ratio of PINNs loss term
        weighting to data-driven loss term weighting.""",
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='ckpts',
        help='Name of path to experiment ckpt folder.',
    )
    parser.add_argument(
        '--ckpt_freq',
        type=int,
        default=5,
        help='The number of epochs between ckpts.',
    )
    parser.add_argument(
        '--test_res',
        type=str,
        default='single',
        choices=['single', 'multi'],
        help="""The testing sets to use during training.
        'single' - test set will have same res and parameters as training
        'multi' - 4 test resolutions w/ same parameters as training

        NOTE: this param doesn't affect model training, simply what stats
        are reported
        """,
    )

    # Data args
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch Size.',
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='darcy',
        choices=[
            'darcy',
            'random',
            'darcy_pdebench',
            'random_fluid',
            'burgers_pdebench',
            'ns_pdebench',
            'incomp_ns_pdebench',
        ],
        help='Training Datasets',
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=32,
        help='Resolution of training data.',
    )
    # NOTE(MS): w/ parsl, the nargs are not all recognized, hack below
    # parser.add_argument(
    #    '--resolution_ratios',
    #    type=float,
    #    default=[0.1, 0.1, 0.1, 0.7],
    #    nargs=4,
    #    help="""Ratio of different size inputs are passed via multiple args
    #        e.g., (.9, .1, .07, .03), which means
    #        that we want 90% of data to be the highest res,
    #        10% to be second highest res, and so on.
    #        """,
    # )
    parser.add_argument(
        '--resolution_ratios',
        type=str,
        default='[0.1, 0.1, 0.1, 0.7]',
        help="""Ratio of different size inputs are passed via multiple args
            e.g., (.9, .1, .07, .03), which means
            that we want 90% of data to be the highest res,
            10% to be second highest res, and so on.
            """,
    )
    parser.add_argument(
        '--filter_lim',
        type=int,
        default=-1,
        help="""lowpass filter limit.
            -1 = no low pass filter applied to data,
            n = freq above n are filtered""",
    )
    parser.add_argument(
        '--downsample_dim',
        type=int,
        default=-1,
        help="""X/Y dim to downsample img to.
            -1 = no downsample.
            n = downsample to n x n img.""",
    )
    parser.add_argument(
        '--initial_steps',
        type=int,
        default=1,
        help="""Number of initial steps in sequence to provide for inference.
            Only relavent for time-varying PDEs like NS or burgers.
            if initial_steps>1, train autoregressively
            """,
    )
    parser.add_argument(
        '--darcy_forcing_term',
        type=float,
        default=1.0,
        choices=[0.01, 0.1, 1.0, 10.0, 100.0],
        help="""Forcing term for Darcy flow.
            PDEBench various forcing terms (beta).
            """,
    )
    parser.add_argument(
        '--burger_viscosity',
        type=float,
        default=0.001,
        choices=[
            0.001,
            0.002,
            0.004,
            0.01,
            0.02,
            0.04,
            0.1,
            0.2,
            0.4,
            1.0,
            2.0,
            4.0,
        ],
        help="""Viscosity for burgers eqation.
            PDEBench various viscosity terms (nu).
            """,
    )
    parser.add_argument(
        '--incomp_ns_viscosity',
        type=float,
        default=0.01,
        choices=[0.01],
        help="""Viscosity for incompressible navier stokes eqation.
            PDEBench fixed viscosity for this dataset.
            """,
    )
    parser.add_argument(
        '--comp_ns_params',
        nargs=6,
        default=['Turb', 1.0, 1e-08, 1e-08, 'periodic', 512],
        help=""" Compressible Navier Stokes from pde Bench
    param should include [type,M,eta,zeta,boundary,resolution] as list""",
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
        choices=['FNO2D', 'FNO1D', 'CROP2D', 'CNO2D'],
        help='Type of model.',
    )
    parser.add_argument(
        '--latent_size',
        type=int,
        default=32,
        help='Latent projection dimention for CROP.',
    )

    args = parser.parse_args()
    # NOTE(MS): parsl workaround
    args.resolution_ratios = eval(args.resolution_ratios)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    seed_everything(args.seed)

    # Get Model
    args.model = get_model(**vars(args))

    # Get DataLoaders
    (args.train_dataloader, args.test_dataloaders) = get_data(**vars(args))

    # Get Loss Function
    args.loss = get_loss(
        args.loss_name,
        args.pinn_loss_weight,
        args.darcy_forcing_term,
        args.burger_viscosity,
        args.incomp_ns_viscosity,
    )

    # Train Model
    train_model(**vars(args))
