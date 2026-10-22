"""
NeuroCodec: Training Script
Main command-line entrypoint for training the NeuroCodec speech compression model.
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from neurocodec import NeuroCodec
from discriminators import CompleteDiscriminator
from losses import NeuroCodecLoss
from dataset import SpeechDataset
from trainer import NeuroCodecTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train NeuroCodec on Speech Audio")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to LibriSpeech or speech audio directory (optional; uses synthetic speech generator if omitted)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Training device ('cpu' or 'cuda')"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume training from"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device

    train_cfg = config.get("training", {})
    audio_cfg = config.get("audio", {})
    model_cfg = config.get("model", {})

    device_str = train_cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Initializing NeuroCodec Training on {device}...")

    # 1. Dataset & DataLoader
    dataset = SpeechDataset(
        audio_dir=args.data_dir,
        sample_rate=audio_cfg.get("sample_rate", 16000),
        segment_length=audio_cfg.get("segment_length", 16000),
        noise_prob=config.get("noise", {}).get("prob", 0.7),
        snr_range=(
            config.get("noise", {}).get("snr_min_db", 5.0),
            config.get("noise", {}).get("snr_max_db", 25.0),
        ),
    )
    batch_size = train_cfg.get("batch_size", 4)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        drop_last=True,
    )
    print(f"Dataset loaded: {len(dataset)} speech samples (Batch size: {batch_size})")

    # 2. Models
    codec = NeuroCodec(
        sample_rate=audio_cfg.get("sample_rate", 16000),
        latent_dim=model_cfg.get("encoder", {}).get("latent_dim", 512),
        hidden_dim=model_cfg.get("encoder", {}).get("hidden_dim", 64),
        num_quantizers=model_cfg.get("rvq", {}).get("num_quantizers", 8),
        codebook_size=model_cfg.get("rvq", {}).get("codebook_size", 1024),
        encoder_strides=tuple(model_cfg.get("encoder", {}).get("strides", [2, 4, 5, 8])),
        decoder_strides=tuple(model_cfg.get("decoder", {}).get("strides", [8, 5, 4, 2])),
        commitment_weight=model_cfg.get("rvq", {}).get("commitment_weight", 0.25),
        quantizer_dropout=model_cfg.get("rvq", {}).get("quantizer_dropout", 0.2),
    )

    discriminator = CompleteDiscriminator(
        periods=tuple(config.get("discriminators", {}).get("multi_period", {}).get("periods", [2, 3, 5, 7, 11]))
    )

    criterion = NeuroCodecLoss(config=config)

    trainer = NeuroCodecTrainer(
        model=codec,
        discriminator=discriminator,
        criterion=criterion,
        config=config,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    epochs = train_cfg.get("epochs", 50)
    save_interval = train_cfg.get("save_checkpoint_interval", 1)

    print("\nStarting Training Loop...")
    for epoch in range(trainer.current_epoch + 1, epochs + 1):
        stats = trainer.train_epoch(dataloader, epoch)
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"G Loss: {stats['g_total']:.4f} | "
            f"D Loss: {stats['d_total']:.4f} | "
            f"STFT Loss: {stats['stft']:.4f} | "
            f"Semantic: {stats['semantic']:.4f} | "
            f"Avg Bitrate: {stats['avg_kbps']:.2f} kbps"
        )

        if epoch % save_interval == 0 or epoch == epochs:
            ckpt_path = trainer.save_checkpoint(epoch)
            print(f"Checkpoint saved: {ckpt_path}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
