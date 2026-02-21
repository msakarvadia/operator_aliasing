#!/usr/bin/env python

# # Demo Notebook
# [paper](https://openreview.net/forum?id=hkF7ZM7fEp&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions) | [blog](https://mansisak.com/operator_aliasing/) | [code](https://github.com/msakarvadia/operator_aliasing)

# In[1]:
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib import colors
from matplotlib import ticker
from torch import nn

data_folder = '/pscratch/sd/m/mansisak/PDEBench/pdebench_data/2D/DarcyFlow/'  # replace with your file path
file_name = '2D_DarcyFlow_beta1.0_Train.hdf5'

FIG_DIR = '../figures'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


# # Utils

# In[2]:


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


# # Data
#
# Download Darcy Dataset from PDEBench: https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download

# In[3]:


import math
import typing
from pathlib import Path

import h5py
import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset
from torchvision.transforms import Compose

# https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py


class DarcyPDEBench(Dataset):
    """Darcy Dataset from PDE Bench."""

    def __init__(
        self,
        filename: str,
        saved_folder: str = '../data/',
        train: bool = True,
        transform: Compose = None,
        resolution_proportions: list[float] = [1, 0, 0, 0],
        **kwargs,
    ):
        """Initialize data.

        :param filename: filename that contains the dataset
        :type filename: STR
        :saved folder: path to PDE bench data
        : train: bool (True = train, False=Test)
        : transforms: Any torch transforms (we will define downsampling/filtering transforms)
        : resolution_proportions: must sum to 1. Fraction of dataset for resolutions [128,64,32,16]
            [1,0,0,0] == 100% 128 res
            [0,0.5,0.5,0] == 50% 64, 50% 32
        support img dimentions: 128, 64, 32, 16 (highest to lowest)
        """
        self.transform = transform

        four = 4
        assert len(resolution_proportions) == four, (
            'Only support 4 img_resolutions, see doc string.'
        )
        assert sum(resolution_proportions) == 1, (
            'All dataset proportions must sum to 1.'
        )
        self.rng = default_rng(seed=0)
        self.batch_size = 128
        test_ratio = 0.1
        num_samples_max = -1

        # Define path to files
        self.model_inputs = []
        self.labels = []
        root_path = Path(Path(saved_folder).resolve()) / filename
        # preprocessing for train/test sets and
        # ...initializing/shuffling data indexes
        with h5py.File(root_path, 'r') as f:
            # num of data samples
            num_samples_max = f['tensor'].shape[0]

            # list of data idxs
            data_idx = np.arange(0, num_samples_max)
            # num of test samples
            test_idx = int(num_samples_max * test_ratio)
            if train:
                first_batch_idx = test_idx
                last_batch_idx = -1
                self.num_samples = num_samples_max - test_idx
            else:
                first_batch_idx = 0
                last_batch_idx = test_idx
                self.num_samples = test_idx
            print(f'{self.num_samples=}')
            # grab data indexs
            self.data_idxs = data_idx[first_batch_idx:last_batch_idx]
            # shuffle indexes
            self.rng.shuffle(self.data_idxs)

        for res_factor, ratio in enumerate(resolution_proportions):
            reduced_resolution = 2**res_factor
            with h5py.File(root_path, 'r') as f:
                # number of points in this resolution set
                res_idx = int(self.num_samples * ratio)
                # sort all indexes
                set_indexes = np.sort(self.data_idxs[:res_idx])
                # u: label
                label = np.array(
                    f['tensor'][
                        set_indexes,
                        :,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ],
                    dtype=np.float32,
                )

                # batch, time, x,...
                _data = np.array(f['nu'], dtype=np.float32)
                # nu: input
                model_input = _data[
                    set_indexes,
                    None,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ]

                self.model_inputs.append(torch.tensor(model_input))
                self.labels.append(torch.tensor(label))

                # remove already used indexes
                self.data_idxs = self.data_idxs[res_idx:]

    def __len__(self) -> int:
        """Returns len of dataset.

        Recall this is a pre-batched dataset, so we return
        number of batches.
        """
        total_batches = 0
        for _set_idx, res_set in enumerate(self.model_inputs):
            num_batches_in_set = math.ceil(len(res_set) / self.batch_size)
            total_batches += num_batches_in_set
        return total_batches

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        """Get single sample at idx."""
        # iterate through all resoulution sets to find batch
        for _set_idx, res_set in enumerate(self.model_inputs):
            num_batches_in_set = math.ceil(len(res_set) / self.batch_size)
            if batch_idx >= num_batches_in_set:
                batch_idx -= num_batches_in_set
            else:
                item_idx = int(batch_idx * self.batch_size)
                set_idx = _set_idx
                break

        # return whole batch, not just single datapoint
        sample = {
            'x': self.model_inputs[set_idx][
                item_idx : item_idx + self.batch_size
            ],
            'y': self.labels[set_idx][item_idx : item_idx + self.batch_size],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


# In[4]:


from torch.utils.data import DataLoader


def get_data(
    **data_args: typing.Any,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Get data w/ args."""
    seed = 0
    g = torch.Generator()
    g.manual_seed(seed)

    # set train specific kwarg
    data_args['train'] = True
    train_dataset = DarcyPDEBench(**data_args)

    test_datasets = {}

    # NOTE(MS): single dataset standard
    # intended use: HP search, filter/downsample exp
    test_kwargs = data_args
    test_kwargs['train'] = False
    if test_kwargs['test_res'] == 'single':
        test_dataset = DarcyPDEBench(**test_kwargs)
        test_datasets['test'] = test_dataset

    # multiple test_datasets
    # NOTE(MS): (this may not work w/ multiple downsample/filter regeims)
    # intended us: multi-res training testing
    if test_kwargs['test_res'] == 'multi':
        for res in range(4):
            resolution_ratios = [0, 0, 0, 0]
            resolution_ratios[res] = 1
            test_kwargs['resolution_ratios'] = resolution_ratios
            test_dataset = DarcyPDEBench(**test_kwargs)
            test_datasets[f'test_res_{res}'] = test_dataset

    training_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=g,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    testing_loaders = {}
    for k, test_dataset in test_datasets.items():
        testing_loaders[k] = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            generator=g,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
    return (training_loader, testing_loaders)


# In[5]:


# This is how we graph a train dataset of resolution 128 and test datasets of resolution [128,64,32,16]
train_loader, test_loaders = get_data(
    filename=file_name,
    saved_folder=data_folder,
    train=True,
    transform=None,
    resolution_proportions=[1, 0, 0, 0],
    test_res='multi',
)


# # Model

# In[6]:


from neuralop.models import FNO


def get_fno(n_modes):
    starting_modes = (n_modes, n_modes)
    model = FNO(
        n_modes=starting_modes,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
    )
    return model


get_fno(n_modes=8)


# # Train/Test Utils

# In[7]:


from torch.nn import Module
from torch.optim import AdamW
from tqdm import tqdm


def train_model(model, train_dataloader, test_dataloaders) -> Module:
    """Train a model."""
    # set up trianing args
    epochs = 150
    loss = nn.MSELoss()

    # set up optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.00001,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5
    )
    starting_epoch = 0
    model = model.to(device)

    # train model
    pbar = tqdm(range(starting_epoch, epochs + 1), desc='Training')
    # for epoch in tqdm(range(starting_epoch, epochs + 1), desc="Training"):
    for epoch in pbar:
        train_loss = 0.0
        # timing to exclude potential data loading
        forward_back_time = 0.0
        for _step, batch in enumerate(train_dataloader):
            # NOTE(MS): must remove outer batch dim from dataloader
            # because we pre-batch data due to multi-res training
            input_batch = batch['x'][0].to(device)
            output_batch = batch['y'][0].to(device)
            batch['device'] = device

            optimizer.zero_grad()

            output_pred_batch = model(input_batch)
            loss_f = loss(output_pred_batch, output_batch)

            loss_f.backward()
            optimizer.step()
            train_loss += loss_f.item()
        train_loss /= len(train_dataloader)
        scheduler.step()
        # test model
        test_dict = test_model(
            model,
            test_dataloaders,
            device,
        )
        pbar_str = f'{train_loss:.1}'
        for key in test_dict.keys():
            pbar_str += f' {key}={test_dict[key]:.1}'
        pbar.set_postfix(train_loss=pbar_str)

    return model.to('cpu')


def test_model(
    model: Module,
    test_dataloaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> dict[str, float]:
    """Test model."""
    test_dict = {}
    loss = nn.MSELoss()
    with torch.no_grad():
        model.eval()
        for test_label, test_dataloader in test_dataloaders.items():
            test_relative_l2 = 0.0
            for _step, batch in enumerate(test_dataloader):
                # NOTE(MS): must remove outer batch dim from dataloader
                # because we pre-batch data due to multi-res training
                input_batch = batch['x'][0].to(device)
                output_batch = batch['y'][0].to(device)
                batch['device'] = device

                output_pred_batch = model(input_batch)
                loss_f = loss(
                    output_pred_batch,
                    output_batch,
                )

                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(test_dataloader)
            test_dict[test_label] = test_relative_l2
    return test_dict


# # Zero-shot super-resolution
#
# Train a model @ resolution 16, test at resoluitons [16,32,64,128].
# Visualize the average spectra across all 4 test sets.

# In[8]:


model = get_fno(n_modes=8)
train_dataloader, test_dataloaders = get_data(
    filename=file_name,
    saved_folder=data_folder,
    train=True,
    transform=None,
    resolution_proportions=[0, 0, 0, 1],
    test_res='multi',
)

model = train_model(model, train_dataloader, test_dataloaders)


# In[9]:


# fig, axs = plt.subplots(3, 4, figsize=(8, 8))

fig = plt.figure(layout='constrained', figsize=(8, 3.5))
subfigs = fig.subfigures(2, 1, wspace=0.07)  # three rows

img_axs = subfigs[0].subplots(1, 4, sharey=False)
spectrum_axs = subfigs[1].subplots(
    2, 4, sharey='row', sharex='col', gridspec_kw={'height_ratios': [1, 1]}
)
model = model.to(device)
model = model.eval()
zero_shot_aucs = []

for x_idx, res in enumerate([16, 32, 64, 128]):
    resolution_ratio = [0, 0, 0, 0]
    resolution_ratio[-(x_idx + 1)] = 1
    resolutions = [16, 32, 64, 128]  # [128, 64, 32, 16]
    dataset = DarcyPDEBench(
        filename=file_name,
        saved_folder=data_folder,
        train=False,
        transform=None,
        resolution_proportions=resolution_ratio,
    )
    # sample = dataset[0]
    sample = {
        'x': torch.concat([dataset[i]['x'] for i in range(len(dataset))]),
        'y': torch.concat([dataset[i]['y'] for i in range(len(dataset))]),
    }
    # grab a single input from the last batch
    model_input = sample['x'].to(device)
    with torch.no_grad():
        model_output = model(x=model_input)
    ground_truth = sample['y']
    residuals = model_output - ground_truth.to(device)

    # convert to plotable form
    model_in = model_input[0, 0, :, :].cpu().numpy()
    output = model_output[0, 0, :, :].cpu().numpy()
    label = ground_truth[0, 0, :, :].cpu().numpy()

    # get spectrums
    label_spectrum = get_energy_curve(ground_truth)
    pred_spectrum = get_energy_curve(model_output)
    residual_spectrum = get_energy_curve(residuals)

    img = output  # - label
    norm = colors.LogNorm(img.mean() + 0.5 * img.std(), img.max(), clip='True')
    img_axs[x_idx].imshow(img, norm=norm, cmap='bwr')

    img_axs[x_idx].get_yaxis().set_visible(False)
    img_axs[x_idx].get_xaxis().set_visible(False)
    subfigs[0].supylabel('Example Prediction')

    spectrum_axs[0, x_idx].set_yscale('log')
    spectrum_axs[1, x_idx].set_yscale('log')
    x = np.linspace(1, len(label_spectrum), len(label_spectrum))
    spectrum_axs[0, x_idx].plot(
        x, label_spectrum, label='Ground Truth', color='blue', ls=':'
    )
    spectrum_axs[0, x_idx].plot(
        x, pred_spectrum, label='Model Prediction', color='orange'
    )
    spectrum_axs[0, x_idx].fill_between(
        x, label_spectrum, pred_spectrum, color='red', alpha=0.1
    )
    spectrum_axs[0, x_idx].axvline(
        x=8, color='gray', linestyle='--', label='Train Limit'
    )
    spectrum_axs[1, x_idx].axvline(x=8, color='gray', linestyle='--')
    normalized_residuals = residual_spectrum / label_spectrum
    spectrum_axs[1, x_idx].plot(
        x,
        normalized_residuals,
        label='Normalized Residual',
        color='red',
    )
    spectrum_axs[0, 0].set_ylabel('Energy')
    spectrum_axs[1, 0].set_ylabel('Energy')
    spectrum_axs[1, x_idx].xaxis.set_major_locator(
        ticker.MaxNLocator(integer=True, nbins=4)
    )
    spectrum_axs[1, x_idx].set_xlabel('Frequency')
    zero_shot_aucs.append(abs(np.trapz(x, normalized_residuals).item()))


handles, labels = spectrum_axs[0, x_idx].get_legend_handles_labels()
_handles, _labels = spectrum_axs[1, x_idx].get_legend_handles_labels()
handles = handles + _handles
labels = labels + _labels
fig.legend(
    handles=handles,
    labels=labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    fancybox=False,
    shadow=False,
    ncol=9,
)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/zero_{device}.pdf', bbox_inches='tight', dpi=300)
plt.show()


# # Resolution Interpolation

# # Information Extrapolation

# # Multi-Resolution Training
#
# Train a model @ resolutions [16,32,64,128] (each resolution comprises 25% of dataset), test at resoluitons [16,32,64,128].
# Visualize the average spectra across all 4 test sets.

# In[ ]:


model = get_fno(n_modes=64)
train_dataloader, test_dataloaders = get_data(
    filename=file_name,
    saved_folder=data_folder,
    train=True,
    transform=None,
    resolution_proportions=[0.25, 0.25, 0.25, 0.25],
    test_res='multi',
)

model = train_model(model, train_dataloader, test_dataloaders)


# In[ ]:


# fig, axs = plt.subplots(3, 4, figsize=(8, 8))

fig = plt.figure(layout='constrained', figsize=(8, 3.5))
subfigs = fig.subfigures(2, 1, wspace=0.07)  # three rows

img_axs = subfigs[0].subplots(1, 4, sharey=False)
spectrum_axs = subfigs[1].subplots(
    2, 4, sharey='row', sharex='col', gridspec_kw={'height_ratios': [1, 1]}
)
model = model.to(device)
model = model.eval()
multi_res_aucs = []

for x_idx, res in enumerate([16, 32, 64, 128]):
    resolution_ratio = [0, 0, 0, 0]
    resolution_ratio[-(x_idx + 1)] = 1
    resolutions = [16, 32, 64, 128]  # [128, 64, 32, 16]
    dataset = DarcyPDEBench(
        filename=file_name,
        saved_folder=data_folder,
        train=False,
        transform=None,
        resolution_proportions=resolution_ratio,
    )
    # sample = dataset[0]
    sample = {
        'x': torch.concat([dataset[i]['x'] for i in range(len(dataset))]),
        'y': torch.concat([dataset[i]['y'] for i in range(len(dataset))]),
    }
    # grab a single input from the last batch
    model_input = sample['x'].to(device)
    with torch.no_grad():
        model_output = model(x=model_input)
    ground_truth = sample['y']
    residuals = model_output - ground_truth.to(device)

    # convert to plotable form
    model_in = model_input[0, 0, :, :].cpu().numpy()
    output = model_output[0, 0, :, :].cpu().numpy()
    label = ground_truth[0, 0, :, :].cpu().numpy()

    # get spectrums
    label_spectrum = get_energy_curve(ground_truth)
    pred_spectrum = get_energy_curve(model_output)
    residual_spectrum = get_energy_curve(residuals)

    img = output  # - label
    norm = colors.LogNorm(img.mean() + 0.5 * img.std(), img.max(), clip='True')
    img_axs[x_idx].imshow(img, norm=norm, cmap='bwr')

    img_axs[x_idx].get_yaxis().set_visible(False)
    img_axs[x_idx].get_xaxis().set_visible(False)
    subfigs[0].supylabel('Example Prediction')

    spectrum_axs[0, x_idx].set_yscale('log')
    spectrum_axs[1, x_idx].set_yscale('log')
    x = np.linspace(1, len(label_spectrum), len(label_spectrum))
    spectrum_axs[0, x_idx].plot(
        x, label_spectrum, label='Ground Truth', color='blue', ls=':'
    )
    spectrum_axs[0, x_idx].plot(
        x, pred_spectrum, label='Model Prediction', color='orange'
    )
    spectrum_axs[0, x_idx].fill_between(
        x, label_spectrum, pred_spectrum, color='red', alpha=0.1
    )
    spectrum_axs[0, x_idx].axvline(
        x=8, color='gray', linestyle='--', label='Train Limit'
    )
    spectrum_axs[1, x_idx].axvline(x=8, color='gray', linestyle='--')
    normalized_residuals = residual_spectrum / label_spectrum
    spectrum_axs[1, x_idx].plot(
        x,
        normalized_residuals,
        label='Normalized Residual',
        color='red',
    )
    multi_res_aucs.append(abs(np.trapz(x, normalized_residuals).item()))
    spectrum_axs[0, 0].set_ylabel('Energy')
    spectrum_axs[1, 0].set_ylabel('Energy')
    spectrum_axs[1, x_idx].xaxis.set_major_locator(
        ticker.MaxNLocator(integer=True, nbins=4)
    )
    spectrum_axs[1, x_idx].set_xlabel('Frequency')


handles, labels = spectrum_axs[0, x_idx].get_legend_handles_labels()
_handles, _labels = spectrum_axs[1, x_idx].get_legend_handles_labels()
handles = handles + _handles
labels = labels + _labels
fig.legend(
    handles=handles,
    labels=labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    fancybox=False,
    shadow=False,
    ncol=9,
)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/multi_{device}.pdf', bbox_inches='tight', dpi=300)
plt.show()


# # Multi-Res Training vs. Zero-shot Super Resolution

# In[ ]:


res = ['16', '32', '64', '128']
w = 0.4
x = np.array([1, 2, 3, 4])

plt.bar(x + w / 2, zero_shot_aucs, w, label='Zero-shot')
plt.bar(x - w / 2, multi_res_aucs, w, label='Multi-Res')

plt.yscale('log')

plt.xticks(x, res)
plt.ylabel('AUC of Normlaized Residual Spectra (Log Scale)')
plt.title(f'Device: {device}')
plt.legend()
plt.savefig(
    f'{FIG_DIR}/multi_v_zero_{device}.pdf', bbox_inches='tight', dpi=300
)
plt.show()

# Use zip() to combine the lists, and then convert the result to a list
zipped_data = list(zip(res, zero_shot_aucs, multi_res_aucs))

# Create the DataFrame, specifying column names
df = pd.DataFrame(zipped_data, columns=['Res', 'Zero', 'Multi'])
df.to_csv(f'{device}_aucs.csv')
