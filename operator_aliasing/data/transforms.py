"""Transforms to apply to data prior to training."""

from __future__ import annotations

import torch
import torch.nn.functional as f

from operator_aliasing.utils import filter_batch
from operator_aliasing.utils import get_1d_low_pass_filter
from operator_aliasing.utils import get_2d_low_pass_filter


class LowpassFilter:
    """Lowpass filter the image.

    Args:
        filter_limit: frequencies > filter_lim excluded
    """

    def __init__(self, filter_limit: int, img_size: int, n_dim: int) -> None:
        """Initialize filter transform.

        filter_limit: number of frequencies to keep
        n_dim: number of spatial dimentions
        """
        assert isinstance(filter_limit, int)
        assert isinstance(img_size, int)
        self.filter_limit = filter_limit
        self.img_size = img_size
        self.n_dim = n_dim

        # assert that filter limit is less than half img_size
        assert self.filter_limit <= self.img_size // 2

        # get filter
        if n_dim == 1:
            self.filter = get_1d_low_pass_filter(
                self.filter_limit, self.img_size
            )
        else:
            self.filter = get_2d_low_pass_filter(
                self.filter_limit, self.img_size
            )

    def __call__(
        self, sample: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply transform."""
        model_input, label = sample['x'], sample['y']

        # apply no filter
        if self.filter_limit == -1:
            return sample

        filter_input = filter_batch(self.filter, model_input, self.n_dim)
        filter_label = filter_batch(self.filter, label, self.n_dim)

        sample['x'] = filter_input
        sample['y'] = filter_label
        return sample


class DownSample:
    """Downsize image.

    Args:
        out_size: x/y dim of downsampled obj
    """

    def __init__(self, out_size: int, n_dim: int) -> None:
        """Initialize downsample transform.

        outsize: resized spatial dim
        n_dim: number of spatial dims
        """
        assert isinstance(out_size, int)
        self.out_size = out_size
        self.n_dim = n_dim

    def __call__(
        self, sample: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply transform."""
        model_input, label = sample['x'], sample['y']

        # apply no downsample
        if self.out_size == -1:
            return sample

        time_dim = model_input.shape[0]
        shape = tuple([self.out_size] * self.n_dim)
        antialias = self.n_dim > 1
        mode = 'bicubic' if self.n_dim > 1 else 'linear'
        if time_dim == 1:
            # 4 dim: time x channels x Xdim x Ydim
            # 3 dim: channels x Xdim x Ydim

            # NOTE(MS): because we downsample 1 img at a time
            # we must first exapand the batch dim
            # then collapse the batch dim to be compatible
            # w/ interpolate & also dataloader
            model_input = model_input.unsqueeze(0)
            label = label.unsqueeze(0)

        downsample_input = f.interpolate(
            model_input,
            size=shape,
            mode=mode,
            antialias=antialias,
        )

        downsample_label = f.interpolate(
            label,
            size=shape,
            mode=mode,
            antialias=antialias,
        )

        sample['x'] = downsample_input
        sample['y'] = downsample_label
        return sample
