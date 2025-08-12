"""Utility functions for experiments."""

from __future__ import annotations

import random
import typing

import numpy as np
import torch
from neuralop.data.datasets.tensor_dataset import TensorDataset
from torch.nn import Module

# def get_energy_curve(data: torch.Tensor) -> torch.Tensor:
#    """Energy Calculation used by Liangzhao."""
#    cpu = 'cpu'
#
#    # https://arxiv.org/pdf/2404.07200v2
#    # Calculates the energy spectrum curve for a batch of 2D arrays.
#    # data: The input data, expected to be a tensor of shape (B, H, W),
#    # where B is the batch size, and H, W are the height and width
#    #           of the 2D arrays.
#
#    data_f = (
#        torch.square(
#            torch.abs(torch.fft.fftshift(torch.fft.fft2(data), dim=(1, 2)))
#        )
#        / (data.size()[-1] * data.size()[-2]) ** 2
#    )
#    # The first division by (H*W) is a normalization
#    # related to Parseval's theorem.
#    # The second division by (H*W) makes the sum
#    # of the final `data_f` equal to
#    # the Mean Squared Value (or average energy per pixel)
#    # of the original array.
#
#    n_mode = (data.size()[-1] + 1) // 2
#    center = data.size()[-1] // 2
#
#    f_energy = []
#
#    # NOTE(MS): this implementation is only good for odd length images
#    for ii in range(n_mode):
#        d_f = data_f[
#            :, center - ii : center + ii + 1, center - ii : center + ii + 1
#        ]
#        d_f[:, 1:-1, 1:-1] = 0
#        # Sum the energy within the isolated shell over
#        # the spatial dimensions (H, W).
#        f_energy.append(torch.sum(d_f, dim=(1, 2)))
#
#    f_energy = torch.stack(f_energy, dim=-1)
#    # Compute the mean energy curve across the entire batch (dimension 0).
#    f_energy = torch.mean(f_energy, dim=0).to(cpu)
#    return f_energy


def autoregressive_inference(
    initial_steps: int,
    model: Module,
    input_batch: torch.tensor,
    output_batch: torch.tensor,
) -> torch.tensor:
    """Autoregressive training loop for time-varying PDE training."""
    # adapted from: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/train.py

    img_size = input_batch.shape[-1]
    batch_size = input_batch.shape[0]
    t_train = output_batch.shape[1]  # number of time steps
    shape: typing.Any = (batch_size, -1)
    for _dim in range(model.n_dim):
        shape += (img_size,)

    all_model_preds = []
    for _t in range(initial_steps, t_train):
        # Model run
        model_input = torch.reshape(input_batch, shape)
        output_pred_batch = model(model_input)
        all_model_preds.append(output_pred_batch)

        # Concatenate the prediction at the current
        # time step to be used as input for the next time step
        input_batch = torch.cat(
            (input_batch[:, 1:, ...], output_pred_batch.unsqueeze(dim=1)),
            dim=1,
        )

    # stack all model preds
    all_model_preds = torch.stack(all_model_preds, dim=1)
    return all_model_preds


def get_model_preds(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    initial_steps: int,
    # data_transform: DataProcessor,
) -> torch.Tensor:
    """Return model predictions."""
    model_preds = []
    model = model.to(device)
    with torch.no_grad():
        for _idx, sample in enumerate(test_loader):  # resolution 128
            model_input = sample['x'][0].to(device)
            model_output = sample['y'][0].to(device)
            with torch.no_grad():
                if initial_steps > 1:
                    out = autoregressive_inference(
                        initial_steps, model, model_input, model_output
                    )
                else:
                    out = model(model_input)
                model_preds.append(out)
    return torch.cat(model_preds)


def generate_wavenumbers(n: int = 6) -> torch.tensor:
    """Generate the wavenumbers."""
    # n = 7  # Size of the square array
    center = n // 2  # Center of the array
    array = np.zeros((n, n), dtype=int)

    # Fill values based on distance from the center
    for i in range(n):
        for j in range(n):
            distance = max(abs(center - i), abs(center - j))
            array[i, j] = distance

    # For even-sized arrays, ensure the center area avoids 0 directly
    if n % 2 == 0:
        for i in range(n):
            for j in range(n):
                if i + j >= n:
                    array[i, j] += 1
    return array


def generate_wavenumbers_1d(n: int = 6) -> torch.tensor:
    """Generate the wavenumbers."""
    # n = 7  # Size of the square array
    center = n // 2  # Center of the array
    array = np.zeros((n), dtype=int)

    # Fill values based on distance from the center
    for i in range(n):
        distance = abs(center - i)
        array[i] = distance

    # For even-sized arrays, ensure the center area avoids 0 directly
    if n % 2 == 0:
        for i in range(n):
            if i >= center:
                array[i] += 1

    return array


def get_energy_curve(
    data: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Calculate 2d spectrum of data.

    data dim: batch x time x X x Y
    """
    signal = data.cpu()
    batch_size = signal.shape[0]
    time_points = signal.shape[1]
    n_observations = signal.shape[-1]
    signal = signal.view(
        batch_size, time_points, n_observations, n_observations
    )

    if normalize:
        signal = torch.fft.fft2(signal, norm='ortho')
    else:
        signal = torch.fft.rfft2(
            signal, s=(n_observations, n_observations), norm='backward'
        )

    # center FFT
    centered_fft_signal = torch.fft.fftshift(signal)

    # compute energy
    energy = centered_fft_signal.abs() ** 2

    # define wavenumbers
    wave_numbers = generate_wavenumbers(n=n_observations)
    max_wavenumber = n_observations // 2

    spectrum = torch.zeros((batch_size, time_points, n_observations // 2))
    for j in range(1, max_wavenumber + 1):
        ind = torch.where(torch.tensor(wave_numbers) == j)
        spectrum[:, :, j - 1] = energy[:, :, ind[0], ind[1]].sum(dim=-1)

    time_avg_spectrum = spectrum.mean(dim=1)
    batch_avg_spectrum = time_avg_spectrum.mean(dim=0)
    return batch_avg_spectrum


def get_energy_curve_1d(
    data: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Calculate 1d spectrum of data."""
    signal = data.cpu()
    batch_size = signal.shape[0]
    time_points = signal.shape[1]
    n_observations = signal.shape[-1]
    signal = signal.view(batch_size, time_points, n_observations)

    if normalize:
        signal = torch.fft.fftn(signal, norm='ortho')
    else:
        signal = torch.fft.rfftn(
            signal,
            s=(n_observations),
            norm='backward',
        )

    # center FFT
    centered_fft_signal = torch.fft.fftshift(signal)

    # compute energy
    energy = centered_fft_signal.abs() ** 2

    # define wavenumbers
    wave_numbers = generate_wavenumbers_1d(n=n_observations)
    max_wavenumber = n_observations // 2

    spectrum = torch.zeros((batch_size, time_points, n_observations // 2))
    for j in range(1, max_wavenumber + 1):
        ind = torch.where(torch.tensor(wave_numbers) == j)
        spectrum[:, :, j - 1] = energy[:, :, ind[0]].sum(dim=-1)

    time_avg_spectrum = spectrum.mean(dim=1)
    batch_avg_spectrum = time_avg_spectrum.mean(dim=0)
    return batch_avg_spectrum


def get_1d_low_pass_filter(f: int, s: int) -> torch.Tensor:
    """Return's low pass filter at limit f and size s.

    f : the frequency limit (only allow freq < f)
    s : dimention of image
    """
    if f > s // 2:
        raise Exception(f'Max frequency of image is {s//2=}, so lower f')
    # 1 = keep, 0 = get rid of
    # central of shifted FFT image is lowest freqs
    wave_numbers = generate_wavenumbers_1d(s)

    filt = torch.where(torch.tensor(wave_numbers) < f, 1, 0)
    return filt.unsqueeze(dim=0)


def get_2d_low_pass_filter(f: int, s: int) -> torch.Tensor:
    """Return's low pass filter at limit f and size s.

    f : the frequency limit (only allow freq < f)
    s : dimention of image
    """
    if f > s // 2:
        raise Exception(f'Max frequency of image is {s//2=}, so lower f')
    # 1 = keep, 0 = get rid of
    # central of shifted FFT image is lowest freqs
    wave_numbers = generate_wavenumbers(s)

    filt = torch.where(torch.tensor(wave_numbers) < f, 1, 0)
    return filt.unsqueeze(dim=0)


def filter_batch(
    filt: torch.tensor, batch: torch.tensor, ndim: int
) -> torch.Tensor:
    """Apply (low-pass) filter to batch.

    filt:
            filter for the batch
            (already centered i.e., torch.fft.fftshift)
            dim: (channel x X_dim x Y_dim) 2D
    batch:
            input batch
            dim: (batch x channel x X_dim x Y_dim ) 2D
    ndim:
            number of spatial dimentions

    """
    dim: typing.Any = (-2, -1)
    if ndim == 1:
        dim = -1
    # fft batch
    batch_fourier = torch.fft.fftn(batch, dim=dim)

    # center batch
    fourier_centered = torch.fft.fftshift(batch_fourier)

    # apply filter
    filtered_batch = fourier_centered * filt

    # convert batch back to spatial domain
    filtered_batch = torch.real(
        torch.fft.ifftn(torch.fft.ifftshift(filtered_batch), dim=dim)
    )

    return filtered_batch


def lowpass_filter_dataloader(
    img_size: int,
    filter_limit: int,
    original_dataloader: torch.utils.data.Dataloader,
    device: torch.device,
    ndim: int,
) -> torch.utils.data.Dataloader:
    """Filter all data in dataloader.

    img_size: assume square img, len of x axis
    filter_limit: frequencies > filter_lim excluded
    original_dataloader: data to filter
    ndim: number of spatial dimentions
    """
    x_data = []
    y_data = []

    # get filter
    if ndim == 1:
        low_pass_filter = get_1d_low_pass_filter(filter_limit, img_size)
    else:
        low_pass_filter = get_2d_low_pass_filter(filter_limit, img_size)
    low_pass_filter = torch.tensor(low_pass_filter, device=device)

    # filter x and y
    for _idx, sample in enumerate(original_dataloader):  # resolution 128
        # x_data.append(sample['x'])
        x_data.append(
            filter_batch(low_pass_filter, sample['x'].to(device), ndim)
        )
        y_data.append(
            filter_batch(low_pass_filter, sample['y'].to(device), ndim)
        )

    # reformat filtered data into dataloader
    filter_x = torch.cat(x_data).to('cpu')
    filter_y = torch.cat(y_data).to('cpu')

    filtered_dataset = TensorDataset(filter_x, filter_y)

    dataloader = torch.utils.data.DataLoader(
        filtered_dataset,
        batch_size=original_dataloader.batch_size,
        num_workers=original_dataloader.num_workers,
        pin_memory=original_dataloader.pin_memory,
        persistent_workers=original_dataloader.persistent_workers,
    )

    return dataloader


def seed_everything(seed: int) -> None:
    """Setting seeds for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seeding dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
