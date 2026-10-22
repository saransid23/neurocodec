"""
NeuroCodec: Adversarial Discriminators
Implements Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD)
with intermediate feature matching for high-fidelity speech synthesis.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodDiscriminator(nn.Module):
    """
    Sub-discriminator for Multi-Period Discriminator.
    Reshapes 1D waveform into 2D (B, 1, T/period, period) to capture periodic speech structures.
    """

    def __init__(
        self,
        period: int,
        channels: Tuple[int, ...] = (32, 128, 512, 1024, 1024),
        kernel_size: int = 5,
        stride: int = 3,
    ):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList()

        in_ch = 1
        for ch in channels:
            self.convs.append(
                nn.Conv2d(
                    in_ch,
                    ch,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=((kernel_size - 1) // 2, 0),
                )
            )
            in_ch = ch

        self.post_conv = nn.Conv2d(in_ch, 1, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap: List[torch.Tensor] = []
        B, C, T = x.shape

        # Pad T to be multiple of period
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            T = x.shape[-1]

        # Reshape to (B, C, T // period, period)
        x = x.view(B, C, T // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.post_conv(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    Ensemble of Period Discriminators with prime periods [2, 3, 5, 7, 11].
    """

    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period=p) for p in periods]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits: List[torch.Tensor] = []
        fmaps: List[List[torch.Tensor]] = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


class ScaleDiscriminator(nn.Module):
    """
    Sub-discriminator for Multi-Scale Discriminator operating on 1D waveforms.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
                nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20),
                nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20),
                nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20),
                nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20),
                nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
            ]
        )
        self.post_conv = nn.Conv1d(1024, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap: List[torch.Tensor] = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.post_conv(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """
    Ensemble of discriminators operating at original, 2x downsampled, and 4x downsampled audio.
    """

    def __init__(self, scales: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList([ScaleDiscriminator() for _ in range(scales)])
        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits: List[torch.Tensor] = []
        fmaps: List[List[torch.Tensor]] = []
        curr = x
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                curr = self.pooling(curr)
            logit, fmap = disc(curr)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


class CompleteDiscriminator(nn.Module):
    """
    Unified Multi-Period and Multi-Scale Discriminator.
    """

    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11), scales: int = 3):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods=periods)
        self.msd = MultiScaleDiscriminator(scales=scales)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Evaluates both MPD and MSD.
        Returns:
            all_logits: Combined list of discriminator logits.
            all_fmaps: Combined list of intermediate feature representations.
        """
        mpd_logits, mpd_fmaps = self.mpd(x)
        msd_logits, msd_fmaps = self.msd(x)

        return mpd_logits + msd_logits, mpd_fmaps + msd_fmaps
