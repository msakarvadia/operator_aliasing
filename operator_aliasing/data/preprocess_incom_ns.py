"""We preprocess the incompressible navier stokes dataest from PDEBench."""

from __future__ import annotations

import glob

import h5py
import torch

from operator_aliasing.train.pinn_losses import finite_difference_2d


def curl(xfield: torch.tensor, yfield: torch.tensor) -> torch.Tensor:
    """Curl operator.

    xfield: x_direction commponent (e.g., velocity or force)
    yfield: y_direction commponent (e.g., velocity or force)
    """
    xfield_dy = finite_difference_2d(u=xfield, d=1, axis='dy')
    yfield_dx = finite_difference_2d(u=yfield, d=1, axis='dx')
    return yfield_dx - xfield_dy


# NOTE: download PDEBench NS incompressible data
# https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download
# record where you downloaded the data:
pdebench_ns_incomp_dir = '../../../PDEBench/pdebench_data/2D/NS_incom/'

with h5py.File(
    f'{pdebench_ns_incomp_dir}/full_data_merge.h5', mode='w'
) as h5fw:
    row1 = 0

    h5fw.require_dataset(
        'Vx',
        dtype='f',
        shape=(1096, 20, 510, 510),
        maxshape=(1096, 20, 510, 510),
    )
    h5fw.require_dataset(
        'Vy',
        dtype='f',
        shape=(1096, 20, 510, 510),
        maxshape=(1096, 20, 510, 510),
    )
    h5fw.require_dataset(
        'vorticity',
        dtype='f',
        shape=(1096, 20, 510, 510),
        maxshape=(1096, 20, 510, 510),
    )
    h5fw.require_dataset(
        'force_curl',
        dtype='f',
        shape=(1096, 510, 510),
        maxshape=(1096, 510, 510),
    )
    for h5name in glob.glob(
        f'{pdebench_ns_incomp_dir}/ns_incom_inhom_2d_512*.h5'
    ):
        h5fr = h5py.File(h5name, 'r')
        Vx = h5fr['velocity'][:, ::50, :, :, 0]  # bs, time, x, y, 2(x,y vel)
        Vy = h5fr['velocity'][:, ::50, :, :, 1]  # bs, time, x, y, 2(x,y vel)
        Fx = h5fr['force'][..., 0]  # bs, x, y, 2(x,y force)
        Fy = h5fr['force'][..., 1]  # bs, x, y, 2(x,y force)
        vorticity = curl(Vx, Vy)
        force_curl = curl(Fx, Fy)

        batch_size = Vx.shape[0]

        h5fw['Vx'][row1 : row1 + batch_size, ...] = Vx[:, ..., 1:-1, 1:-1]
        h5fw['Vy'][row1 : row1 + batch_size, ...] = Vy[:, ..., 1:-1, 1:-1]
        h5fw['vorticity'][row1 : row1 + batch_size, ...] = vorticity[:, ...]
        h5fw['force_curl'][row1 : row1 + batch_size, ...] = force_curl[:, ...]

        row1 += batch_size
        print(row1)
