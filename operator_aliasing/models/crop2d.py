"""CROP lifting framework adapted to FNO.

https://openreview.net/pdf?id=J9FgrqOOni

https://github.com/wenhangao21/ICLR25-CROP/blob/main/original_code_and_trained_models/Sec5_1_NS_with_low_Reynolds/CROP/CROP.py
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

################################################################
# fourier layer
################################################################


class SpectralWeights(nn.Module):
    """Spectral Weights."""

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int
    ) -> None:
        """Initialize."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        dtype = torch.cfloat
        self.kernel_size_Y = 2 * modes1 - 1
        self.kernel_size_X = modes2
        self.W = nn.ParameterDict(
            {
                'y0_modes': torch.nn.Parameter(
                    torch.empty(
                        in_channels, out_channels, modes1 - 1, 1, dtype=dtype
                    )
                ),
                'yposx_modes': torch.nn.Parameter(
                    torch.empty(
                        in_channels,
                        out_channels,
                        self.kernel_size_Y,
                        self.kernel_size_X - 1,
                        dtype=dtype,
                    )
                ),
                '00_modes': torch.nn.Parameter(
                    torch.empty(
                        in_channels, out_channels, 1, 1, dtype=torch.float
                    )
                ),
            }
        )
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self) -> None:
        """Reset Params."""
        for v in self.W.values():
            nn.init.kaiming_uniform_(v, a=math.sqrt(5))

    def get_weight(self) -> None:
        """Get Weights."""
        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        self.weights = torch.cat(
            [
                self.W['y0_modes'],
                self.W['00_modes'].cfloat(),
                self.W['y0_modes'].flip(dims=(-2,)).conj(),
            ],
            dim=-2,
        )
        self.weights = torch.cat([self.weights, self.W['yposx_modes']], dim=-1)
        self.weights = self.weights.view(
            self.in_channels,
            self.out_channels,
            self.kernel_size_Y,
            self.kernel_size_X,
        )


class SpectralConv2d(nn.Module):
    """Spectral convolution."""

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int
    ) -> None:
        """2D Fourier layer. It does FFT, linear transform, and Inverse FFT."""
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.spectral_weight = SpectralWeights(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
        )
        self.get_weight()

    def get_weight(self) -> None:
        """Get Weights."""
        self.spectral_weight.get_weight()
        self.weights = self.spectral_weight.weights

    # Complex multiplication
    def compl_mul2d(
        self, input_mat: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Complex mat mul."""
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) ->
        # (batch, out_channel, x,y)
        return torch.einsum('bixy,ioxy->boxy', input_mat, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Inference."""
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_y = (
            (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0)
            .nonzero()
            .item()
        )
        self.get_weight()
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[
            ...,
            (freq0_y - self.modes1 + 1) : (freq0_y + self.modes1),
            : self.modes2,
        ]
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[
            ...,
            (freq0_y - self.modes1 + 1) : (freq0_y + self.modes1),
            : self.modes2,
        ] = self.compl_mul2d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfft2(
            torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1))
        )
        return x


class MLP(nn.Module):
    """MLP."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: int
    ) -> None:
        """Initialize."""
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Inference."""
        x = self.mlp1(x)
        x = f.gelu(x)
        x = self.mlp2(x)
        return x


class CropToLatentSize(nn.Module):
    """Project to latent size."""

    def __init__(self, in_size: int, out_size: int) -> None:
        """Initialize."""
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.temp_size = min(in_size, out_size)

    def forward(self, u1: torch.Tensor) -> torch.Tensor:
        """Forward Inference."""
        batch, channel, _, _ = u1.shape
        fu1 = torch.fft.rfft2(u1, norm='ortho')
        fu1_recover = torch.zeros(
            (batch, channel, self.out_size, self.out_size // 2 + 1),
            dtype=torch.complex64,
            device=u1.device,
        )
        fu1_recover[:, :, : self.temp_size // 2, : self.temp_size // 2 + 1] = (
            fu1[:, :, : self.temp_size // 2, : self.temp_size // 2 + 1]
        )
        fu1_recover[
            :, :, -self.temp_size // 2 :, : self.temp_size // 2 + 1
        ] = fu1[:, :, -self.temp_size // 2 :, : self.temp_size // 2 + 1]
        # Inverse FFT and scaling
        u1_recover = torch.fft.irfft2(fu1_recover, norm='ortho') * (
            self.out_size / self.in_size
        )
        return u1_recover


class CROPFNO2d(nn.Module):
    """CROP Model."""

    def __init__(
        self,
        modes: tuple[int, int],
        width: int,
        # in_size: int,
        latent_size: int,
        time_steps: int,
    ):
        """The overall network. It contains 4 layers of the Fourier layer.

        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output
            space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps
            + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        super().__init__()

        self.modes1 = modes[0]
        self.modes2 = modes[1]
        self.width = width
        # self.in_size = in_size
        self.latent_size = latent_size
        self.padding = 8  # pad the domain if input is non-periodic
        self.time_steps = time_steps

        # input channel is 12: the solution of the previous 10 timesteps
        # + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.p = nn.Conv2d(self.time_steps, self.width, 1)
        self.conv0 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv1 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv2 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv3 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2
        )
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(
            self.width, 1, self.width * 4
        )  # output channel is 1: u(x, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Inference.

        x: input dim (batch_size, time_steps, x_dim, y_dim)
        """
        self.in_size = x.shape[-1]
        self.CROP_to_latent = CropToLatentSize(self.in_size, self.latent_size)
        self.CROP_back = CropToLatentSize(self.latent_size, self.in_size)
        # x = x.permute(0, 3, 1, 2)
        # print('Pre projection: ', x.shape)
        x = self.CROP_to_latent(x)
        # print('Post projection: ', x.shape)
        x = self.p(x)

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = f.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = f.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = f.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # print('Pre crop back: ', x.shape)
        x = self.CROP_back(x)
        # print('Post crop back: ', x.shape)

        x = self.q(x)
        # x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(
        self, shape: tuple[int, int, int], device: torch.device
    ) -> torch.tensor:
        """Get Grid."""
        batchsize, size_x, size_y = shape[0], shape[-2], shape[-1]
        gridx = torch.tensor(
            np.linspace(0, 1, size_x), dtype=torch.float, device=device
        )
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(
            [batchsize, 1, size_y, 1]
        )
        gridy = torch.tensor(
            np.linspace(0, 1, size_y), dtype=torch.float, device=device
        )
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(
            [batchsize, size_x, 1, 1]
        )
        return torch.cat((gridx, gridy), dim=-1)
