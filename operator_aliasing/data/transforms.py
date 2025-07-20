"""Transforms to apply to data prior to training."""

from __future__ import annotations

import torch
import torch.nn.functional as f

from operator_aliasing.utils import filter_batch
from operator_aliasing.utils import get_2d_low_pass_filter


class LowpassFilter2D:
    """Lowpass filter the image.

    Args:
        filter_limit: frequencies > filter_lim excluded
    """

    def __init__(self, filter_limit: int, img_size: int) -> None:
        """Initialize filter transform."""
        assert isinstance(filter_limit, int)
        assert isinstance(img_size, int)
        self.filter_limit = filter_limit
        self.img_size = img_size

        # assert that filter limit is less than half img_size
        assert self.filter_limit <= self.img_size // 2

        # get filter
        self.filter = get_2d_low_pass_filter(self.filter_limit, self.img_size)

    def __call__(
        self, sample: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply transform."""
        model_input, label = sample['x'], sample['y']

        # apply no filter
        if self.filter_limit == -1:
            return {'x': model_input, 'y': label}

        filter_input = filter_batch(self.filter, model_input)
        filter_label = filter_batch(self.filter, label)

        return {'x': filter_input, 'y': filter_label}


class DownSample:
    """Downsize image.

    Args:
        out_size: x/y dim of downsampled obj
    """

    def __init__(self, out_size: int) -> None:
        """Initialize downsample transform."""
        assert isinstance(out_size, int)
        self.out_size = out_size

    def __call__(
        self, sample: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply transform."""
        model_input, label = sample['x'], sample['y']

        # apply no downsample
        if self.out_size == -1:
            return {'x': model_input, 'y': label}

        shape = (self.out_size, self.out_size)
        time_dim = model_input.shape[0]
        if time_dim != 1:
            # 4 dim: time x channels x Xdim x Ydim
            # 3 dim: channels x Xdim x Ydim
            # shape = (model_input.shape[1],) + shape
            label_time_dim = label.shape[0]
            channel_dim = model_input.shape[1]
            spatial_dim = model_input.shape[2]

            # collapse time x channel dim:
            print(f'{model_input.shape=}')
            model_input = torch.reshape(
                model_input, (time_dim * channel_dim, spatial_dim, spatial_dim)
            )
            label = torch.reshape(
                label, (label_time_dim * channel_dim, spatial_dim, spatial_dim)
            )

        # NOTE(MS): because we downsample 1 img at a time
        # we must first exapand the batch dim
        # then collapse the batch dim to be compatible
        # w/ interpolate & also dataloader
        downsample_input = f.interpolate(
            model_input.unsqueeze(0),
            size=shape,
            mode='bicubic',
            antialias=True,
        )[0]

        downsample_label = f.interpolate(
            label.unsqueeze(0),
            size=shape,
            mode='bicubic',
            antialias=True,
        )[0]

        if time_dim != 1:
            # 4 dim: time x channels x Xdim x Ydim
            # uncollapse time x channel dim:
            downsample_input = torch.reshape(
                downsample_input,
                (time_dim, channel_dim, self.out_size, self.out_size),
            )
            downsample_label = torch.reshape(
                downsample_label,
                (label_time_dim, channel_dim, self.out_size, self.out_size),
            )

        print(f'{downsample_input.shape=}')
        return {'x': downsample_input, 'y': downsample_label}
