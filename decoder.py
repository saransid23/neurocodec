"""
NeuroCodec: Convolutional Audio Decoder
Upsamples latent representations back into 16 kHz high-fidelity speech waveforms.
"""

from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import ResidualUnit1d


class DecoderBlock(nn.Module):
    """
    Upsampling block consisting of ConvTranspose1d followed by
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
        padding = (kernel_size - stride) // 2
        output_padding = (kernel_size - stride) % 2
        self.upsample = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.res_units = nn.ModuleList(
            [ResidualUnit1d(out_channels, dilation=d) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        for unit in self.res_units:
            x = unit(x)
        return x


class AudioDecoder(nn.Module):
    """
    Convolutional Speech Decoder for NeuroCodec.
    Transforms quantized latent representations (B, 512, T_frames) back into
    raw 16 kHz audio waveforms (B, 1, T).
    Upsampling ratio = 8 * 5 * 4 * 2 = 320.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        out_channels: int = 1,
        strides: Tuple[int, ...] = (8, 5, 4, 2),
        dilations: Tuple[int, ...] = (1, 3, 9),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.strides = strides
        self.hop_length = math.prod(strides)  # 320

        # Initial projection from latent_dim
        self.init_conv = nn.Conv1d(
            latent_dim, hidden_dim, kernel_size=7, stride=1, padding=3
        )

        # Upsampling stages
        blocks: List[nn.Module] = []
        curr_dim = hidden_dim
        # Stage channel decay: 512 -> 256 -> 128 -> 64
        stage_dims = [
            max(hidden_dim // (2**i), 64) for i in range(1, len(strides) + 1)
        ]

        for s, next_dim in zip(strides, stage_dims):
            blocks.append(
                DecoderBlock(
                    in_channels=curr_dim,
                    out_channels=next_dim,
                    stride=s,
                    dilations=dilations,
                )
            )
            curr_dim = next_dim

        self.blocks = nn.ModuleList(blocks)

        # Final projection to audio channels
        self.final_conv = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(curr_dim, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),  # Bounded to [-1.0, 1.0]
        )

    def forward(
        self, z_q: torch.Tensor, target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            z_q: Quantized latent tensor (B, latent_dim, T_frames).
            target_length: Desired waveform length (samples) to match input audio.
        Returns:
            audio: Reconstructed waveform (B, 1, T).
        """
        x = self.init_conv(z_q)
        for block in self.blocks:
            x = block(x)
        audio = self.final_conv(x)

        if target_length is not None:
            curr_len = audio.shape[-1]
            if curr_len > target_length:
                audio = audio[..., :target_length]
            elif curr_len < target_length:
                audio = F.pad(audio, (0, target_length - curr_len))

        return audio
