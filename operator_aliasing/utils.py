"""Utility functions for experiments."""

from __future__ import annotations

import neuralop
import torch


def get_energy_curve(data: torch.Tensor) -> torch.Tensor:
    """Energy Calculation used by Liangzhao."""
    cpu = 'cpu'

    # https://arxiv.org/pdf/2404.07200v2
    # Calculates the energy spectrum curve for a batch of 2D arrays.
    # data: The input data, expected to be a tensor of shape (B, H, W),
    # where B is the batch size, and H, W are the height and width
    #           of the 2D arrays.

    data_f = (
        torch.square(
            torch.abs(torch.fft.fftshift(torch.fft.fft2(data), dim=(1, 2)))
        )
        / (data.size()[-1] * data.size()[-2]) ** 2
    )
    # The first division by (H*W) is a normalization
    # related to Parseval's theorem.
    # The second division by (H*W) makes the sum
    # of the final `data_f` equal to
    # the Mean Squared Value (or average energy per pixel)
    # of the original array.

    n_mode = (data.size()[-1] + 1) // 2
    center = data.size()[-1] // 2

    f_energy = []

    for ii in range(n_mode):
        d_f = data_f[
            :, center - ii : center + ii + 1, center - ii : center + ii + 1
        ]
        d_f[:, 1:-1, 1:-1] = 0
        # Sum the energy within the isolated shell over
        # the spatial dimensions (H, W).
        f_energy.append(torch.sum(d_f, dim=(1, 2)))

    f_energy = torch.stack(f_energy, dim=-1)
    # Compute the mean energy curve across the entire batch (dimension 0).
    f_energy = torch.mean(f_energy, dim=0).to(cpu)
    return f_energy


def get_model_preds(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    data_transform: neuralop.data.transforms.DataProcessor,
) -> torch.Tensor:
    """Return model predictions."""
    model_preds = []
    for _idx, sample in enumerate(test_loader):  # resolution 128
        # NOTE(MS): might want to pass data transform in here
        model_input = data_transform.preprocess(sample)
        with torch.no_grad():
            out = model(**model_input)
            model_preds.append(out)
    return torch.cat(model_preds)
