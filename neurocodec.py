"""
NeuroCodec: Unified Neural Audio Codec
Combines Convolutional Encoder, Residual Vector Quantization (RVQ),
AI-Driven Importance Network for Dynamic Bitrate Allocation, and Convolutional Decoder.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import AudioEncoder
from decoder import AudioDecoder
from quantization import ResidualVectorQuantizer


class ImportanceNetwork(nn.Module):
    """
    Lightweight temporal neural network that predicts the perceptual importance
    of each audio frame and dynamically allocates RVQ quantizers.
    Higher acoustic complexity / phonetic content receives up to 8 quantizers,
    while silence or stationary background regions are allocated fewer quantizers (e.g. 1-2).
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        max_quantizers: int = 8,
        min_quantizers: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_quantizers = max_quantizers
        self.min_quantizers = min_quantizers

        # Temporal convolution and projection
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(hidden_dim // 2, max_quantizers, kernel_size=1),
        )

        # Global importance scalar head for visualization
        self.score_head = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        manual_active_levels: Optional[int] = None,
        bitrate_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Latent continuous features (B, input_dim, T).
            temperature: Softmax / sigmoid sharpening temperature.
            manual_active_levels: If provided (1..8), overrides dynamic allocation.
            bitrate_scale: Multiplier for importance scores (0.1 to 2.0) for interactive control.

        Returns:
            quantizer_mask: Soft or straight-through mask (B, max_quantizers, T) in [0, 1].
            active_counts: Discrete active quantizer count per frame (B, T) in [min, max].
            importance_scores: Normalized scalar importance score per frame (B, T) in [0, 1].
        """
        B, D, T = z.shape
        device = z.device

        # If user explicitly fixes quantizer count:
        if manual_active_levels is not None:
            k = max(min(manual_active_levels, self.max_quantizers), self.min_quantizers)
            mask = torch.zeros(B, self.max_quantizers, T, device=device)
            mask[:, :k, :] = 1.0
            counts = torch.full((B, T), k, dtype=torch.long, device=device)
            # Use RMS of z as surrogate importance
            importance_scores = torch.norm(z, dim=1)
            importance_scores = (
                importance_scores / (importance_scores.max(dim=1, keepdim=True)[0] + 1e-6)
            ).clamp(0.0, 1.0)
            return mask, counts, importance_scores

        # Feature processing
        h1 = F.elu(self.net[0](z))
        h2 = F.elu(self.net[2](h1))

        # Importance scalar score per frame (B, T)
        scores = self.score_head(h2).squeeze(1)  # (B, T) in [0, 1]
        scaled_scores = (scores * bitrate_scale).clamp(0.0, 1.0)

        # Quantizer activation logits per stage (B, max_quantizers, T)
        logits = self.net[4](h2)

        # Progressive activation probabilities: cumulative sigmoid thresholding
        # Each quantizer k is active if frame priority is high enough
        probs = torch.sigmoid(logits / temperature)

        # Guarantee the first min_quantizers levels are always active (out-of-place)
        min_mask = torch.arange(self.max_quantizers, device=device).view(1, -1, 1) < self.min_quantizers
        probs = torch.where(min_mask, torch.ones_like(probs), probs)

        if self.training:
            # Straight-Through Gumbel or STE binary mask for backpropagation
            hard_mask = (probs > 0.5).float()
            quantizer_mask = probs + (hard_mask - probs).detach()
        else:
            # During inference, binary hard thresholding modulated by bitrate_scale
            quantizer_mask = (probs * bitrate_scale >= 0.5).float()
            quantizer_mask = torch.where(min_mask, torch.ones_like(quantizer_mask), quantizer_mask)

        # Discrete count per frame = sum of active quantizers
        active_counts = quantizer_mask.sum(dim=1).long().clamp(
            self.min_quantizers, self.max_quantizers
        )

        return quantizer_mask, active_counts, scaled_scores


class NeuroCodec(nn.Module):
    """
    Unified High-Fidelity Speech Neural Audio Codec (16 kHz).
    Integrates:
      - Convolutional Encoder (downsamples 320x -> 50 Hz frames)
      - AI-Driven Importance Network (frame-adaptive quantizer activation)
      - 8-stage Residual Vector Quantizer (1024 codebooks x 512 dimensions)
      - Convolutional Decoder (upsamples 320x back to audio)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        in_channels: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 512,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        encoder_strides: Tuple[int, ...] = (2, 4, 5, 8),
        decoder_strides: Tuple[int, ...] = (8, 5, 4, 2),
        commitment_weight: float = 0.25,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.bits_per_codebook = int(math.log2(codebook_size))  # 10 bits

        # 1. Encoder
        self.encoder = AudioEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            strides=encoder_strides,
        )
        self.hop_length = self.encoder.hop_length  # 320
        self.frame_rate = sample_rate / self.hop_length  # 50 Hz

        # 2. Importance Network
        self.importance_net = ImportanceNetwork(
            input_dim=latent_dim,
            hidden_dim=256,
            max_quantizers=num_quantizers,
            min_quantizers=1,
        )

        # 3. Residual Vector Quantizer
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=latent_dim,
            commitment_weight=commitment_weight,
            quantizer_dropout=quantizer_dropout,
        )

        # 4. Decoder
        self.decoder = AudioDecoder(
            latent_dim=latent_dim,
            hidden_dim=latent_dim,
            out_channels=in_channels,
            strides=decoder_strides,
        )

    def forward(
        self,
        x: torch.Tensor,
        manual_quantizers: Optional[int] = None,
        bitrate_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Full forward pass: Encode -> Importance Allocation -> RVQ -> Decode.

        Args:
            x: Audio tensor of shape (B, 1, T) or (B, T).
            manual_quantizers: Optional fixed number of quantizers (1..8).
            bitrate_scale: Dynamic multiplier for adaptive bitrate selection.

        Returns:
            Dictionary containing:
              - 'reconstructed': Reconstructed audio (B, 1, T)
              - 'z': Continuous latent features (B, latent_dim, T_frames)
              - 'quantized': Quantized latents (B, latent_dim, T_frames)
              - 'indices': Codebook tokens (B, num_quantizers, T_frames)
              - 'importance_mask': Active quantizer mask (B, num_quantizers, T_frames)
              - 'active_counts': Integer quantizer count per frame (B, T_frames)
              - 'importance_scores': Frame importance scores (B, T_frames)
              - 'commitment_loss': Scalar RVQ commitment loss
              - 'codebook_loss': Scalar RVQ codebook loss
              - 'bitrate_kbps': Scalar average bitrate in kbps
              - 'per_frame_kbps': Per-frame bitrate tensor in kbps
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        orig_length = x.shape[-1]

        # 1. Encode
        z = self.encoder(x)

        # 2. AI Importance Allocation
        mask, active_counts, importance_scores = self.importance_net(
            z,
            manual_active_levels=manual_quantizers,
            bitrate_scale=bitrate_scale,
        )

        # 3. Residual Vector Quantization with Mask
        vq_out = self.quantizer(z, quantizer_mask=mask)
        z_q = vq_out["quantized"]
        indices = vq_out["indices"]

        # 4. Decode
        reconstructed = self.decoder(z_q, target_length=orig_length)

        # 5. Bitrate analytics
        # Bitrate per frame = active_quantizers * 10 bits * frame_rate / 1000
        per_frame_kbps = (
            active_counts.float() * self.bits_per_codebook * self.frame_rate / 1000.0
        )
        avg_bitrate_kbps = per_frame_kbps.mean()

        return {
            "reconstructed": reconstructed,
            "z": z,
            "quantized": z_q,
            "indices": indices,
            "importance_mask": mask,
            "active_counts": active_counts,
            "importance_scores": importance_scores,
            "commitment_loss": vq_out["commitment_loss"],
            "codebook_loss": vq_out["codebook_loss"],
            "bitrate_kbps": avg_bitrate_kbps,
            "per_frame_kbps": per_frame_kbps,
        }

    def encode(
        self,
        x: torch.Tensor,
        manual_quantizers: Optional[int] = None,
        bitrate_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compress raw audio to quantized discrete tokens with frame-level metadata.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        mask, active_counts, importance_scores = self.importance_net(
            z,
            manual_active_levels=manual_quantizers,
            bitrate_scale=bitrate_scale,
        )
        vq_out = self.quantizer(z, quantizer_mask=mask)

        per_frame_kbps = (
            active_counts.float() * self.bits_per_codebook * self.frame_rate / 1000.0
        )

        return {
            "indices": vq_out["indices"],
            "active_counts": active_counts,
            "importance_scores": importance_scores,
            "importance_mask": mask,
            "avg_bitrate_kbps": per_frame_kbps.mean().item(),
            "per_frame_kbps": per_frame_kbps,
            "original_length": x.shape[-1],
        }

    def decode(
        self,
        indices: torch.Tensor,
        active_counts: Optional[torch.Tensor] = None,
        quantizer_mask: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decompress discrete codebook tokens into audio waveform.
        """
        if quantizer_mask is None and active_counts is not None:
            B, T = active_counts.shape
            device = indices.device
            layer_idx = torch.arange(self.num_quantizers, device=device).view(1, -1, 1)
            quantizer_mask = (layer_idx < active_counts.unsqueeze(1)).float()

        z_q = self.quantizer.decode_from_indices(
            indices, quantizer_mask=quantizer_mask
        )
        audio = self.decoder(z_q, target_length=target_length)
        return audio
