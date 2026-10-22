"""
NeuroCodec: Residual Vector Quantization (RVQ)
Implements 8-stage Residual Vector Quantization with codebook size 1024 and dimension 512.
Supports adaptive dynamic quantizer activation per frame.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Single-layer Vector Quantizer using Euclidean distance and
    Straight-Through Estimator (STE) for backpropagation.
    """

    def __init__(
        self,
        codebook_size: int = 1024,
        codebook_dim: int = 512,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight

        # Codebook embedding: (codebook_size, codebook_dim)
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        # Uniform initialization across unit sphere
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input continuous latent tensor of shape (B, codebook_dim, T).
        Returns:
            z_q: Quantized latent with STE gradient (B, codebook_dim, T).
            indices: Codebook indices (B, T).
            commitment_loss: ||z - sg(q)||^2.
            codebook_loss: ||sg(z) - q||^2.
        """
        # Transpose to (B, T, codebook_dim) for distance computation
        z_t = z.permute(0, 2, 1).contiguous()
        B, T, D = z_t.shape
        flat_z = z_t.view(-1, D)

        # Pairwise Euclidean distances: (flat_z - emb)^2 = z^2 + emb^2 - 2 z * emb
        # Using memory-efficient expansion
        emb_weights = self.embedding.weight  # (codebook_size, D)
        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(emb_weights**2, dim=1)
            - 2 * torch.matmul(flat_z, emb_weights.t())
        )

        indices_flat = torch.argmin(distances, dim=1)
        indices = indices_flat.view(B, T)

        # Lookup quantized embeddings
        quantized_flat = self.embedding(indices_flat)
        quantized = quantized_flat.view(B, T, D).permute(0, 2, 1).contiguous()

        # Compute VQ losses
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())

        # Straight-Through Estimator: pass gradients directly from z_q to z
        z_q = z + (quantized - z).detach()

        return z_q, indices, commitment_loss, codebook_loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode integer indices (B, T) to quantized embeddings (B, codebook_dim, T).
        """
        quantized = self.embedding(indices)  # (B, T, codebook_dim)
        return quantized.permute(0, 2, 1).contiguous()


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) comprising multiple sequential quantizers.
    Default: 8 quantizers, codebook size 1024, dimension 512.
    Supports:
      - Static quantizer activation (e.g. fixed 1..8 quantizers)
      - Dynamic per-frame quantizer activation predicted by the Importance Network
      - Quantizer dropout during training to improve resilience of lower levels
    """

    def __init__(
        self,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 512,
        commitment_weight: float = 0.25,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.quantizer_dropout = quantizer_dropout

        self.layers = nn.ModuleList(
            [
                VectorQuantizer(
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    commitment_weight=commitment_weight,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(
        self,
        z: torch.Tensor,
        num_active_quantizers: Optional[Union[int, torch.Tensor]] = None,
        quantizer_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: Latent feature map of shape (B, codebook_dim, T).
            num_active_quantizers:
                - None: use all quantizers (or apply dropout if training)
                - int: activate exactly first N quantizers
                - torch.Tensor: shape (B, T) with integer counts per frame in [1, num_quantizers]
            quantizer_mask: Optional explicit binary/soft mask of shape (B, num_quantizers, T).

        Returns:
            Dictionary containing:
              - 'quantized': Aggregated quantized representation (B, codebook_dim, T)
              - 'indices': Quantizer codebook indices (B, num_quantizers, T)
              - 'commitment_loss': Total commitment loss across active layers
              - 'codebook_loss': Total codebook loss across active layers
              - 'active_counts': Frame-level or batch-level active quantizer count
        """
        B, D, T = z.shape
        device = z.device

        # Determine active quantizers or build frame mask
        if quantizer_mask is None:
            if isinstance(num_active_quantizers, int):
                # Fixed active quantizers across all frames
                k_val = min(max(num_active_quantizers, 1), self.num_quantizers)
                mask = torch.zeros(B, self.num_quantizers, T, device=device)
                mask[:, :k_val, :] = 1.0
            elif isinstance(num_active_quantizers, torch.Tensor):
                # Per-frame dynamic count (B, T)
                # Create mask where layer k is active if k < count
                # layer_indices: (1, num_quantizers, 1)
                layer_idx = torch.arange(self.num_quantizers, device=device).view(1, -1, 1)
                counts = num_active_quantizers.unsqueeze(1)  # (B, 1, T)
                mask = (layer_idx < counts).float()  # (B, num_quantizers, T)
            else:
                # Default: use all, but support random quantizer dropout during training
                if self.training and self.quantizer_dropout > 0.0:
                    # Randomly sample active levels between 1 and num_quantizers
                    rand_k = torch.randint(
                        1, self.num_quantizers + 1, (1,), device=device
                    ).item()
                    mask = torch.zeros(B, self.num_quantizers, T, device=device)
                    mask[:, :rand_k, :] = 1.0
                else:
                    mask = torch.ones(B, self.num_quantizers, T, device=device)
        else:
            mask = quantizer_mask

        residual = z
        quantized_out = torch.zeros_like(z)
        all_indices: List[torch.Tensor] = []
        total_commitment_loss = torch.tensor(0.0, device=device)
        total_codebook_loss = torch.tensor(0.0, device=device)

        for i, layer in enumerate(self.layers):
            layer_mask = mask[:, i : i + 1, :]  # (B, 1, T)

            q_k, idx_k, c_loss, cb_loss = layer(residual)
            all_indices.append(idx_k)

            # Apply mask to this quantizer level
            q_k_effective = q_k * layer_mask
            quantized_out = quantized_out + q_k_effective

            # Residual for next quantizer is updated only where active
            residual = residual - q_k_effective

            # Weight loss by mask presence directly without python conditional
            mask_weight = layer_mask.mean()
            total_commitment_loss = total_commitment_loss + c_loss * mask_weight
            total_codebook_loss = total_codebook_loss + cb_loss * mask_weight

        indices_stacked = torch.stack(all_indices, dim=1)  # (B, num_quantizers, T)

        return {
            "quantized": quantized_out,
            "indices": indices_stacked,
            "commitment_loss": total_commitment_loss,
            "codebook_loss": total_codebook_loss,
            "mask": mask,
        }

    def decode_from_indices(
        self,
        indices: torch.Tensor,
        num_quantizers: Optional[int] = None,
        quantizer_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reconstruct continuous latent representation from quantized indices.

        Args:
            indices: Tensor of shape (B, num_quantizers, T)
            num_quantizers: Optional integer limiting decoding to first N stages
            quantizer_mask: Optional binary/continuous mask (B, num_quantizers, T)
        Returns:
            z_q: (B, codebook_dim, T)
        """
        B, K, T = indices.shape
        device = indices.device
        k_max = min(K, self.num_quantizers if num_quantizers is None else num_quantizers)

        z_q = torch.zeros(B, self.codebook_dim, T, device=device)

        for i in range(k_max):
            q_k = self.layers[i].decode_indices(indices[:, i, :])  # (B, D, T)
            if quantizer_mask is not None:
                q_k = q_k * quantizer_mask[:, i : i + 1, :]
            z_q = z_q + q_k

        return z_q
