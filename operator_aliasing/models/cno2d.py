"""Public implementation of CNO from camlab."""
# https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/main/CNO2d_vanilla_torch_version/CNO2d.py
# The CNO2d code has been modified from a tutorial featured in the
# ETH Zurich course "AI in the Sciences and Engineering."
# Git page for this course: https://github.com/bogdanraonic3/AI_Science_Engineering

# For up/downsampling, the antialias interpolation functions from the
# torch library are utilized, limiting the ability to design
# your own low-pass filters at present.

# While acknowledging this suboptimal setup,
# the performance of CNO2d remains commendable.
# Additionally, a training script is available,
# offering a solid foundation for personal projects.
from __future__ import annotations

import typing

import torch
import torch.nn.functional as f
from torch import nn

# CNO LReLu activation fucntion
# CNO building block (CNOBlock) → Conv2d - BatchNorm - Activation
# Lift/Project Block (Important for embeddings)
# Residual Block →
#   Conv2d - BatchNorm - Activation - Conv2d - BatchNorm - Skip Connection
# ResNet → Stacked ResidualBlocks (several blocks applied iteratively)


# ---------------------
# Activation Function:
# ---------------------


class CNOLReLUeLu(nn.Module):
    """CNO activation function."""

    def __init__(self, in_size: int, out_size: int) -> None:
        """Initialize activation function."""
        # super(CNOLReLUeLu, self).__init__()
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method for activation function."""
        x = f.interpolate(
            x,
            size=(2 * self.in_size, 2 * self.in_size),
            mode='bicubic',
            antialias=True,
        )
        x = self.act(x)
        x = f.interpolate(
            x,
            size=(self.out_size, self.out_size),
            mode='bicubic',
            antialias=True,
        )
        return x


# --------------------
# CNO Block:
# --------------------


class CNOBlock(nn.Module):
    """CNO Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_size: int,
        out_size: int,
        use_bn: bool = True,
    ) -> None:
        """Initialize function."""
        # super(CNOBlock, self).__init__()
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size

        # -----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        self.convolution = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
        )

        if use_bn:
            self.batch_norm = nn.BatchNorm2d(self.out_channels)
        else:
            self.batch_norm = nn.Identity()
        self.act = CNOLReLUeLu(in_size=self.in_size, out_size=self.out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)


# --------------------
# Lift/Project Block:
# --------------------


class LiftProjectBlock(nn.Module):
    """Lift Projection Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        latent_dim: int = 64,
    ) -> None:
        """Initialize function."""
        # super(LiftProjectBlock, self).__init__()
        super().__init__()

        self.inter_CNOBlock = CNOBlock(
            in_channels=in_channels,
            out_channels=latent_dim,
            in_size=size,
            out_size=size,
            use_bn=False,
        )

        self.convolution = torch.nn.Conv2d(
            in_channels=latent_dim,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        """Forward method."""
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        return x


# --------------------
# Residual Block:
# --------------------


class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, channels: int, size: int, use_bn: bool = True) -> None:
        """Initialize function."""
        # super(ResidualBlock, self).__init__()
        super().__init__()

        self.channels = channels
        self.size = size

        # -----------------------------------------

        # We apply Conv -> BN (optional)
        # -> Activation -> Conv -> BN (optional) -> Skip Connection

        # Up/Downsampling happens inside Activation

        self.convolution1 = torch.nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
        )
        self.convolution2 = torch.nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
        )

        if use_bn:
            self.batch_norm1 = nn.BatchNorm2d(self.channels)
            self.batch_norm2 = nn.BatchNorm2d(self.channels)

        else:
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()

        self.act = CNOLReLUeLu(in_size=self.size, out_size=self.size)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        """Forward method."""
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out


# --------------------
# ResNet:
# --------------------


class ResNet(nn.Module):
    """ResNet backbone."""

    def __init__(
        self, channels: int, size: int, num_blocks: int, use_bn: bool = True
    ) -> None:
        """Initialize function."""
        # super(ResNet, self).__init__()
        super().__init__()

        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(
                ResidualBlock(channels=channels, size=size, use_bn=use_bn)
            )

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        """Forward method."""
        for i in range(self.num_blocks):
            x = self.res_nets[i](x)
        return x


# --------------------
# CNO:
# --------------------


class CNO2d(nn.Module):
    """CNO2d module."""

    def __init__(
        self,
        **model_args: typing.Any,
        # in_dim: int,  # Number of input channels.
        # out_dim: int,  # Number of input channels.
        # size: int,  # Input and Output spatial size (required )
        # n_layers: int,  # Number of (D) or (U) blocks in the network
        # n_res: int = 4,  # Number of (R) blocks per level (except the neck)
        # n_res_neck: int = 4,  # Number of (R) blocks in the neck
        # channel_multiplier: int = 16,  # How the number of channels evolve?
        # use_bn: bool = True,  # Add BN? No  BN in lifting/projection layer
    ) -> None:
        """Initialize function.

        Args and suggested inititalizations:
         in_dim: int,  # Number of input channels.
         out_dim: int,  # Number of input channels.
         size: int,  # Input and Output spatial size (required )
         n_layers: int,  # Number of (D) or (U) blocks in the network
         n_res: int = 4,  # Number of (R) blocks per level (except the neck)
         n_res_neck: int = 4,  # Number of (R) blocks in the neck
         channel_multiplier: int = 16,  # How the number of channels evolve?
         use_bn: bool = True,  # Add BN? No  BN in lifting/projection layer
        """
        in_dim = model_args['in_dim']
        out_dim = model_args['out_dim']
        size = model_args['size']
        n_layers = model_args['n_layers']
        n_res = model_args['n_res']
        n_res_neck = model_args['n_res_neck']
        channel_multiplier = model_args['channel_multiplier']
        use_bn = model_args['use_bn']

        # super(CNO2d, self).__init__()
        super().__init__()

        self.n_layers = int(n_layers)  # Number od (D) & (U) Blocks
        self.lift_dim = (
            channel_multiplier // 2
        )  # Input is lifted to the half of channel_multiplier dimension
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel_multiplier = (
            channel_multiplier  # The growth of the channels
        )

        ######## Num of channels/features - evolution ########

        self.encoder_features = [
            self.lift_dim
        ]  # How the features in Encoder evolve (number of features)
        for i in range(self.n_layers):
            self.encoder_features.append(2**i * self.channel_multiplier)

        self.decoder_features_in = self.encoder_features[
            1:
        ]  # How the features in Decoder evolve (number of features)
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.n_layers):
            self.decoder_features_in[i] = (
                2 * self.decoder_features_in[i]
            )  # Pad the outputs of the resnets (we must multiply by 2 then)

        ######## Spatial sizes of channels - evolution ########

        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.n_layers + 1):
            self.encoder_sizes.append(size // 2**i)
            self.decoder_sizes.append(size // 2 ** (self.n_layers - i))

        ######## Define Lift and Project blocks ########

        self.lift = LiftProjectBlock(
            in_channels=in_dim,
            out_channels=self.encoder_features[0],
            size=size,
        )

        self.project = LiftProjectBlock(
            in_channels=self.encoder_features[0]
            + self.decoder_features_out[-1],
            out_channels=out_dim,
            size=size,
        )

        ######## Define Encoder, ED Linker and Decoder networks ########

        self.encoder = nn.ModuleList(
            [
                (
                    CNOBlock(
                        in_channels=self.encoder_features[i],
                        out_channels=self.encoder_features[i + 1],
                        in_size=self.encoder_sizes[i],
                        out_size=self.encoder_sizes[i + 1],
                        use_bn=use_bn,
                    )
                )
                for i in range(self.n_layers)
            ]
        )

        # After the ResNets are executed, the sizes of
        # encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion = nn.ModuleList(
            [
                (
                    CNOBlock(
                        in_channels=self.encoder_features[i],
                        out_channels=self.encoder_features[i],
                        in_size=self.encoder_sizes[i],
                        out_size=self.decoder_sizes[self.n_layers - i],
                        use_bn=use_bn,
                    )
                )
                for i in range(self.n_layers + 1)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                (
                    CNOBlock(
                        in_channels=self.decoder_features_in[i],
                        out_channels=self.decoder_features_out[i],
                        in_size=self.decoder_sizes[i],
                        out_size=self.decoder_sizes[i + 1],
                        use_bn=use_bn,
                    )
                )
                for i in range(self.n_layers)
            ]
        )

        #### Define ResNets Blocks

        # Here, we define ResNet Blocks.

        # Operator UNet:
        # Outputs of the middle networks are patched (or padded)
        # to corresponding sets of feature maps in the decoder

        self.res_nets = []
        self.n_res = int(n_res)
        self.n_res_neck = int(n_res_neck)

        # Define the ResNet networks (before the neck)
        for layer in range(self.n_layers):
            self.res_nets.append(
                ResNet(
                    channels=self.encoder_features[layer],
                    size=self.encoder_sizes[layer],
                    num_blocks=self.n_res,
                    use_bn=use_bn,
                )
            )

        self.res_net_neck = ResNet(
            channels=self.encoder_features[self.n_layers],
            size=self.encoder_sizes[self.n_layers],
            num_blocks=self.n_res_neck,
            use_bn=use_bn,
        )

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        """Forward method."""
        x = self.lift(x)  # Execute Lift
        skip = []

        # Execute Encoder
        for i in range(self.n_layers):
            # Apply ResNet & save the result
            y = self.res_nets[i](x)
            skip.append(y)

            # Apply (D) block
            x = self.encoder[i](x)

        # Apply the deepest ResNet (bottle neck)
        x = self.res_net_neck(x)

        # Execute Decode
        for i in range(self.n_layers):
            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.n_layers - i](
                    x
                )  # BottleNeck : no cat
            else:
                x = torch.cat(
                    (x, self.ED_expansion[self.n_layers - i](skip[-i])), 1
                )

            # Apply (U) block
            x = self.decoder[i](x)

        # Cat & Execute Projetion
        x = torch.cat((x, self.ED_expansion[0](skip[0])), 1)
        x = self.project(x)

        return x
