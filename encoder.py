"""
NeuroCodec: Convolutional Audio Encoder
Downsamples 16 kHz mono raw waveform into high-dimensional latent representation.
"""

from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUnit1d(nn.Module):
    """
    1D Dilated Residual Block with ELU activations.
    Provides multi-scale receptive field without excessive parameter growth.
    """

    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + res


class EncoderBlock(nn.Module):
    """
    Downsampling block consisting of strided convolution followed by
    multiple dilated residual units.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations: Tuple[int, ...] = (1, 3, 9),
    ):
        super().__init__()
        kernel_size = 2 * stride
        padding = math.ceil((kernel_size - stride) / 2)
        self.downsample = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.res_units = nn.ModuleList(
            [ResidualUnit1d(out_channels, dilation=d) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for unit in self.res_units:
            x = unit(x)
        return x


class AudioEncoder(nn.Module):
    """
    Convolutional Speech Encoder for NeuroCodec.
    Transforms raw 16 kHz mono waveform (B, 1, T) into latent features (B, 512, T_frames).
    Total downsampling ratio = 2 * 4 * 5 * 8 = 320.
    At 16,000 Hz, 320 downsampling yields 50 Hz frame rate (20 ms window).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 512,
        strides: Tuple[int, ...] = (2, 4, 5, 8),
        dilations: Tuple[int, ...] = (1, 3, 9),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.strides = strides
        self.hop_length = math.prod(strides)  # default: 320

        # Initial convolution
        self.init_conv = nn.Conv1d(
            in_channels, hidden_dim, kernel_size=7, stride=1, padding=3
        )

        # Progressive downsampling stages
        blocks: List[nn.Module] = []
        curr_dim = hidden_dim
        # Channel progression: e.g., 64 -> 128 -> 256 -> 512 -> 512
        stage_dims = [
            min(hidden_dim * (2**i), latent_dim) for i in range(1, len(strides) + 1)
        ]

        for s, next_dim in zip(strides, stage_dims):
            blocks.append(
                EncoderBlock(
                    in_channels=curr_dim,
                    out_channels=next_dim,
                    stride=s,
                    dilations=dilations,
                )
            )
            curr_dim = next_dim

        self.blocks = nn.ModuleList(blocks)

        # Final projection to fixed latent_dim (512)
        self.final_proj = nn.Conv1d(curr_dim, latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw audio waveform tensor of shape (B, 1, T) or (B, T).
        Returns:
            z: Continuous latent representation (B, latent_dim, T // hop_length).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        z = self.final_proj(x)
        return z
