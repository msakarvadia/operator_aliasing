# https://github.com/wenhangao21/ICLR25-CROP/blob/main/original_code_and_trained_models/Sec_5_2_all_other_experiments/NS_linearDarcy_Poisson/scr/DON.py # noqa

import math
import typing

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

units = {0: 'B', 1: 'KiB', 2: 'MiB', 3: 'GiB', 4: 'TiB'}


def format_mem(x: int) -> tuple[float, str]:
    """Takes integer 'x' in bytes and returns a number.

    in [0, 1024) and the corresponding unit.
    """
    limit = 1024
    if abs(x) < limit:
        return round(x, 2), 'B'

    scale = math.log2(abs(x)) // 10
    scaled_x = x / limit**scale
    unit = units[int(scale)]

    if int(scaled_x) == scaled_x:
        return int(scaled_x), unit

    # rounding leads to 2 or fewer decimal places, as required
    return round(scaled_x, 2), unit


def format_tensor_size(x: int) -> str:
    """Formatting memory tensor."""
    val, unit = format_mem(x)
    return f'{val}{unit}'


def torch2dgrid(
    num_x: int,
    num_y: int,
    bot: tuple[int, int] = (0, 0),
    top: tuple[int, int] = (1, 1),
) -> torch.tensor:
    """Defining model output grid."""
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = torch.stack([xx, yy], dim=2)
    return mesh


class DenseNet(nn.Module):
    """DenseNet."""

    def __init__(
        self,
        layers: list[int],
        nonlinearity: str | nn.Module,
        out_nonlinearity: nn.Module = None,
        normalize: bool = False,
    ) -> None:
        """Initialize DenseNet."""
        super().__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        nonlinearity_func = nn.ReLU
        if isinstance(nonlinearity, str):
            if nonlinearity == 'tanh':
                nonlinearity_func = nn.Tanh
            elif nonlinearity == 'relu':
                assert nonlinearity_func == nn.ReLU
            else:
                raise ValueError(f'{nonlinearity} is not supported')
        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity_func())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method."""
        for _, layer in enumerate(self.layers):
            x = layer(x)

        return x


class DeepONetCP(nn.Module):
    """DeepONet."""

    def __init__(
        self, branch_layer: list[int], trunk_layer: list[int], model_dim: int
    ) -> None:
        """Initialize deeponet."""
        super().__init__()
        self.branch = DenseNet(branch_layer, nn.ReLU)
        self.trunk = DenseNet(trunk_layer, nn.ReLU)
        self.model_dim = model_dim

    def forward(self, u0: torch.tensor, **kwargs: typing.Any) -> torch.tensor:
        """DeepONet forward method."""
        # only 2d support rn
        batch_size, num_channels, x_dim, y_dim = u0.shape

        # check to make sure input dim matched model dim,
        # if not, downsample input (output dim remains the same)
        if self.model_dim > y_dim:
            raise Exception(
                f'Input dim {y_dim} must be >= than model dim {self.model_dim}'
            )
        if self.model_dim < y_dim:
            factor = y_dim // self.model_dim
            u0 = u0[:, :, ::factor, ::factor]

        # reshape inputs
        u0 = torch.reshape(u0, (batch_size, -1))

        # set output dim
        output_dim = x_dim
        if 'output_dim' in kwargs:
            output_dim = kwargs['output_dim']
        grid = torch2dgrid(output_dim, output_dim, bot=(0, 0), top=(1, 1))
        grid = grid.reshape(-1, 2).to(device)

        a = self.branch(u0)
        # batchsize x width
        b = self.trunk(grid)
        # N x width
        output = torch.einsum('bi,ni->bn', a, b)
        reshaped_output = torch.reshape(
            output, (batch_size, num_channels, output_dim, output_dim)
        )
        return reshaped_output

    def print_size(self) -> int:
        """Print # of model params."""
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(
            f"""Total number of model parameters in DON:
            {nparams} (~{format_tensor_size(nbytes)})"""
        )

        return nparams
