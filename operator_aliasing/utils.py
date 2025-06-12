"""Utility functions for experiments."""

from __future__ import annotations

import numpy as np
import torch
from neuralop.data.datasets.tensor_dataset import TensorDataset

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


def get_model_preds(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    # data_transform: neuralop.data.transforms.DataProcessor,
) -> torch.Tensor:
    """Return model predictions."""
    model_preds = []
    for _idx, sample in enumerate(test_loader):  # resolution 128
        # model_input = data_transform.preprocess(sample)
        model_input = sample
        with torch.no_grad():
            out = model(**model_input)
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


def get_energy_curve(
    data: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Calculate 2d spectrum of data."""
    signal = data.cpu()
    t = signal.shape[0]
    n_observations = signal.shape[-1]
    signal = signal.view(t, n_observations, n_observations)

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

    spectrum = torch.zeros((t, n_observations // 2))
    for j in range(1, max_wavenumber + 1):
        ind = torch.where(torch.tensor(wave_numbers) == j)
        spectrum[:, j - 1] = energy[:, ind[0], ind[1]].sum(dim=1)

    spectrum = spectrum.mean(dim=0)
    return spectrum


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


def filter_batch(filt: torch.tensor, batch: torch.tensor) -> torch.Tensor:
    """Apply (low-pass) filter to batch.

    filt:
            filter for the batch
            (already centered i.e., torch.fft.fftshift)
            dim: (channel x X_dim x Y_dim)
    batch:
            input batch
            dim: (batch x channel x X_dim x Y_dim)
    """
    # fft batch
    batch_fourier = torch.fft.fftn(batch, dim=(-2, -1))

    # center batch
    fourier_centered = torch.fft.fftshift(batch_fourier)

    # apply filter
    filtered_batch = fourier_centered * filt

    # convert batch back to spatial domain
    filtered_batch = torch.real(
        torch.fft.ifftn(torch.fft.ifftshift(filtered_batch), dim=(-1, -2))
    )

    return filtered_batch


def lowpass_filter_dataloader(
    img_size: int,
    filter_limit: int,
    original_dataloader: torch.utils.data.Dataloader,
    device: torch.device,
) -> torch.utils.data.Dataloader:
    """Filter all data in dataloader.

    img_size: assume square img, len of x axis
    filter_limit: frequencies > filter_lim excluded
    original_dataloader: data to filter
    """
    x_data = []
    y_data = []

    # get filter
    low_pass_filter = get_2d_low_pass_filter(filter_limit, img_size)
    low_pass_filter = torch.tensor(low_pass_filter, device=device)

    # filter x and y
    for _idx, sample in enumerate(original_dataloader):  # resolution 128
        x_data.append(filter_batch(low_pass_filter, sample['x'].to(device)))
        y_data.append(filter_batch(low_pass_filter, sample['y'].to(device)))

    # reformat filtered data into dataloader
    filter_x = torch.cat(x_data)
    filter_y = torch.cat(y_data)

    filtered_dataset = TensorDataset(filter_x, filter_y)

    dataloader = torch.utils.data.DataLoader(
        filtered_dataset,
        batch_size=original_dataloader.batch_size,
        num_workers=original_dataloader.num_workers,
        pin_memory=original_dataloader.pin_memory,
        persistent_workers=original_dataloader.persistent_workers,
    )

    return dataloader
