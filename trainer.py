"""
NeuroCodec: Training Engine
Orchestrates adversarial training with AdamW, gradient accumulation,
multi-loss optimization, TensorBoard logging, and checkpoint management.
"""

from typing import Dict, Optional
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from neurocodec import NeuroCodec
from discriminators import CompleteDiscriminator
from losses import NeuroCodecLoss


class NeuroCodecTrainer:
    """
    Handles end-to-end training of the NeuroCodec generator and discriminators.
    """

    def __init__(
        self,
        model: NeuroCodec,
        discriminator: CompleteDiscriminator,
        criterion: NeuroCodecLoss,
        config: Dict,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        train_cfg = config.get("training", {})

        # Device selection
        if device is None:
            dev_str = train_cfg.get("device", "cpu")
            if dev_str == "cuda" and not torch.cuda.is_available():
                dev_str = "cpu"
            self.device = torch.device(dev_str)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.criterion = criterion.to(self.device)

        # Optimization configuration
        lr_g = float(train_cfg.get("learning_rate_g", 2e-4))
        lr_d = float(train_cfg.get("learning_rate_d", 2e-4))
        betas = tuple(train_cfg.get("betas", [0.8, 0.99]))
        weight_decay = float(train_cfg.get("weight_decay", 0.01))

        self.opt_g = torch.optim.AdamW(
            self.model.parameters(), lr=lr_g, betas=betas, weight_decay=weight_decay
        )
        self.opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=lr_d, betas=betas, weight_decay=weight_decay
        )

        # Gradient accumulation
        self.grad_accum_steps = int(train_cfg.get("grad_accum_steps", 4))

        # Checkpoints & TensorBoard
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
        self.log_dir = train_cfg.get("log_dir", "runs/neurocodec")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.global_step = 0
        self.current_epoch = 0

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Runs one complete training epoch over the dataset.
        """
        self.model.train()
        self.discriminator.train()
        self.current_epoch = epoch

        running_losses: Dict[str, float] = {
            "g_total": 0.0,
            "d_total": 0.0,
            "stft": 0.0,
            "adv": 0.0,
            "feat": 0.0,
            "semantic": 0.0,
            "bitrate_reg": 0.0,
            "avg_kbps": 0.0,
        }

        self.opt_g.zero_grad()
        self.opt_d.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        batch_idx = 0

        for noisy_audio, clean_audio in pbar:
            noisy_audio = noisy_audio.to(self.device)
            clean_audio = clean_audio.to(self.device)

            # -------------------------------------------------------------
            # 1. Forward Generator (NeuroCodec)
            # -------------------------------------------------------------
            out = self.model(noisy_audio)
            reconstructed = out["reconstructed"]

            # -------------------------------------------------------------
            # 2. Train Discriminator
            # -------------------------------------------------------------
            real_logits, _ = self.discriminator(clean_audio)
            fake_logits, _ = self.discriminator(reconstructed.detach())

            loss_d = self.criterion.forward_discriminator(real_logits, fake_logits)
            loss_d_scaled = loss_d / self.grad_accum_steps
            loss_d_scaled.backward()

            # -------------------------------------------------------------
            # 3. Train Generator
            # -------------------------------------------------------------
            # Full discriminator pass with gradients for generator
            disc_fake_logits, disc_fake_fmaps = self.discriminator(reconstructed)
            with torch.no_grad():
                disc_real_logits, disc_real_fmaps = self.discriminator(clean_audio)

            vq_dict = {
                "commitment_loss": out["commitment_loss"],
                "codebook_loss": out["codebook_loss"],
                "mask": out["importance_mask"],
            }

            g_losses = self.criterion.forward_generator(
                reconstructed=reconstructed,
                clean_target=clean_audio,
                discriminator_outputs=(disc_fake_logits, disc_fake_fmaps),
                discriminator_reals=(disc_real_logits, disc_real_fmaps),
                vq_outputs=vq_dict,
                importance_scores=out["importance_scores"],
            )

            loss_g = g_losses["total_g_loss"]
            loss_g_scaled = loss_g / self.grad_accum_steps
            loss_g_scaled.backward()

            # Optimizer step on accumulation boundary
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                self.opt_d.step()
                self.opt_g.step()
                self.opt_d.zero_grad()
                self.opt_g.zero_grad()

            # Update running stats
            running_losses["g_total"] += loss_g.item()
            running_losses["d_total"] += loss_d.item()
            running_losses["stft"] += g_losses["loss_stft"].item()
            running_losses["adv"] += g_losses["loss_adv"].item()
            running_losses["feat"] += g_losses["loss_feat"].item()
            running_losses["semantic"] += g_losses["loss_semantic"].item()
            running_losses["bitrate_reg"] += g_losses["loss_bitrate"].item()
            running_losses["avg_kbps"] += out["bitrate_kbps"].item()

            # Tensorboard logging
            if self.global_step % 10 == 0:
                self.writer.add_scalar("Train/Loss_G", loss_g.item(), self.global_step)
                self.writer.add_scalar("Train/Loss_D", loss_d.item(), self.global_step)
                self.writer.add_scalar("Train/Loss_STFT", g_losses["loss_stft"].item(), self.global_step)
                self.writer.add_scalar("Train/Loss_Semantic", g_losses["loss_semantic"].item(), self.global_step)
                self.writer.add_scalar("Train/Avg_Bitrate_kbps", out["bitrate_kbps"].item(), self.global_step)

            pbar.set_postfix(
                G_loss=f"{loss_g.item():.3f}",
                D_loss=f"{loss_d.item():.3f}",
                kbps=f"{out['bitrate_kbps'].item():.2f}",
            )

            batch_idx += 1
            self.global_step += 1

        num_batches = max(len(dataloader), 1)
        epoch_stats = {k: v / num_batches for k, v in running_losses.items()}
        return epoch_stats

    def save_checkpoint(self, epoch: int, path: Optional[str] = None) -> str:
        """
        Saves full model and optimizer checkpoint.
        """
        if path is None:
            path = os.path.join(self.checkpoint_dir, f"neurocodec_epoch_{epoch:03d}.pt")

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "opt_g_state_dict": self.opt_g.state_dict(),
            "opt_d_state_dict": self.opt_d.state_dict(),
            "config": self.config,
        }
        torch.save(state, path)

        # Also save latest pointer
        latest_path = os.path.join(self.checkpoint_dir, "neurocodec_latest.pt")
        torch.save(state, latest_path)
        return path

    def load_checkpoint(self, path: str):
        """
        Loads saved checkpoint to resume training or for evaluation.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        if "opt_g_state_dict" in checkpoint:
            self.opt_g.load_state_dict(checkpoint["opt_g_state_dict"])
        if "opt_d_state_dict" in checkpoint:
            self.opt_d.load_state_dict(checkpoint["opt_d_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        print(f"Loaded checkpoint from {path} (Epoch {self.current_epoch})")
