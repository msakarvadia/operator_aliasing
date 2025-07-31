"""PINNs loss functions."""

from __future__ import annotations

import typing

import numpy as np
import torch
from torch import nn


# https://github.com/neuraloperator/physics_informed/blob/master/train_utils/losses.py#L39
class LpLoss:
    """loss function with rel/abs Lp loss."""

    def __init__(
        self,
        d: int = 2,
        p: int = 2,
        size_average: bool = True,
        reduction: bool = True,
    ) -> None:
        """Initialize LP loss."""
        super()

        # Dimension and Lp-norm type are postive
        assert d > 0
        assert p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute absolute diff."""
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute relative diff."""
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
            self.p,
            1,
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Return loss."""
        return self.rel(x, y)


def fdm_darcy(u: torch.Tensor, a: torch.Tensor, d: int = 1) -> torch.Tensor:
    """Finite Difference Method.

    u = model pred
    a = diffusivity constant (model input)
    """
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = d / (size - 1)
    dy = dx

    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    a = a[:, 1:-1, 1:-1]

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    du = -(auxx + auyy)
    return du


def fdm_burgers(u: torch.Tensor, v: float, d: int = 1) -> torch.Tensor:
    """Finite Difference method.

    u = model pred (across multiple time points)
    v = viscosity
    """
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(-1)

    u = u.reshape(batchsize, nt, nx)
    dt = d / (nt - 1)

    u_h = torch.fft.fft(u, dim=2)
    # Wavenumbers in y-direction
    k_max = nx // 2
    k_x = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1, device=u.device),
            torch.arange(start=-k_max, end=0, step=1, device=u.device),
        ),
        0,
    ).reshape(1, 1, nx)
    ux_h = 2j * np.pi * k_x * u_h
    uxx_h = 2j * np.pi * k_x * ux_h
    ux = torch.fft.irfft(ux_h[:, :, : k_max + 1], dim=2, n=nx)
    uxx = torch.fft.irfft(uxx_h[:, :, : k_max + 1], dim=2, n=nx)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    du = ut + (ux * u - v * uxx)[:, 1:-1, :]
    return du


# NOTE(MS): Reformatting original loss
# to be a torch.nn.module below.
# def PINO_loss(u, u0, v):
#    batchsize = u.size(0)
#    nt = u.size(1)
#    nx = u.size(2)
#
#    u = u.reshape(batchsize, nt, nx)

#    index_t = torch.zeros(nx,).long()
#    index_x = torch.tensor(range(nx)).long()
#    boundary_u = u[:, index_t, index_x]
#    loss_u = F.mse_loss(boundary_u, u0)
#
#    du = fdm_burgers(u, v)[:, :, :]
#    f = torch.zeros(du.shape, device=u.device)
#    loss_f = F.mse_loss(du, f)

#    return loss_u, loss_f


class BurgersDataAndPinnsLoss(nn.Module):
    """Data+Pinns Loss for Burgers flow."""

    def __init__(self, pinn_loss_weight: float, viscosity: float) -> None:
        """Initialize loss.

        viscosity: param of dataset
        https://github.com/neuraloperator/physics_informed/blob/master/train_utils/losses.py#L224
        """
        super().__init__()
        self.L1 = nn.L1Loss()
        self.lploss = LpLoss(size_average=True)
        self.viscosity = viscosity
        self.pinn_loss_weight = pinn_loss_weight

    def forward(
        self,
        model_pred: torch.Tensor,
        ground_truth: torch.Tensor,
        model_input: torch.Tensor,
    ) -> float:
        """Loss calculation.

        model_pred shape: batch_size x time x X_dim
        ground_truth shape: same as model pred
        model input shape: batch_size x initial_steps x X_dim
        """
        data_loss = self.L1(model_pred, ground_truth)

        # NOTE(MS): PINO had initial condition loss
        # for training stability, but we don't need it
        # we train auto-regressively
        # Initical condition loss is cheating, model explicitly seeing IC!!

        # batchsize = model_pred.size(0)
        # nt = model_pred.size(1)
        # nx = model_pred.size(-1)
        # u = model_pred.reshape(batchsize, nt, nx)

        # index_t = torch.zeros(nx).long()
        # index_x = torch.tensor(range(nx)).long()
        # boundary_u = u[:, index_t, index_x]
        # TODO(MS): check whether this is correct
        # u0 = model_input[:, 0, 0, :]
        # boundary_loss = torch.nn.functional.mse_loss(boundary_u, u0)

        du = fdm_burgers(model_pred, self.viscosity)
        f = torch.zeros(du.shape, device=model_pred.device)
        # TODO(MS): these are all MSE (but in darcy we use L1), make consistant
        # pde_loss = torch.nn.functional.mse_loss(du, f)
        # NOTE(MS): standardizing L1 loss for data + physics
        pde_loss = self.L1(du, f)

        # return 0.33 * data_loss + 0.33 * boundary_loss + 0.33 * pde_loss
        return (
            1 - self.pinn_loss_weight
        ) * data_loss + self.pinn_loss_weight * pde_loss


# NOTE(MS): Reformatting original loss
# to be a torch.nn.module below.

# def darcy_loss(u, a):
#    batchsize = u.size(0)
#    size = u.size(1)
#    u = u.reshape(batchsize, size, size)
#    a = a.reshape(batchsize, size, size)
#    lploss = LpLoss(size_average=True)
#    du = fdm_darcy(u, a)
#    f = torch.ones(du.shape, device=u.device)
#    loss_f = lploss.rel(du, f)
#    return loss_f


class L1Loss(nn.Module):
    """Data Loss."""

    def __init__(
        self,
    ) -> None:
        """Initialize loss."""
        super().__init__()
        self.L1 = nn.L1Loss()

    def forward(
        self,
        model_pred: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: typing.Any,
    ) -> float:
        """Loss calculation.

        model_pred shape: batch_size x 1 (no time) x X_dim x Y_dim
        ground_truth shape: same as model pred
        """
        return self.L1(model_pred, ground_truth)


class DarcyDataAndPinnsLoss(nn.Module):
    """Data+Pinns Loss for Darcy flow."""

    def __init__(
        self, pinn_loss_weight: float, darcy_forcing_term: float
    ) -> None:
        """Initialize loss.

        pinn_loss_weight: ratio of data vs. pinn loss
        """
        super().__init__()
        self.L1 = nn.L1Loss()
        self.lploss = LpLoss(size_average=True)
        self.pinn_loss_weight = pinn_loss_weight
        self.darcy_forcing_term = darcy_forcing_term

    def forward(
        self,
        model_pred: torch.Tensor,
        ground_truth: torch.Tensor,
        model_input: torch.Tensor,
    ) -> float:
        """Loss calculation.

        model_pred shape: batch_size x 1 (no time) x X_dim x Y_dim
        ground_truth shape: same as model pred
        """
        data_loss = self.L1(model_pred, ground_truth)

        batchsize = model_pred.size(0)
        size = ground_truth.size(-1)
        u = model_pred.reshape(batchsize, size, size)
        a = model_input.reshape(batchsize, size, size)
        du = fdm_darcy(u, a)
        f = torch.ones(du.shape, device=u.device) * self.darcy_forcing_term
        # NOTE(MS): standardizing L1 loss for data + physics
        # pinn_loss = self.lploss.rel(du, f)
        pinn_loss = self.L1(du, f)

        return (
            1 - self.pinn_loss_weight
        ) * data_loss + self.pinn_loss_weight * pinn_loss
