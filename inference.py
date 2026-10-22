"""
NeuroCodec: Inference & Real-Time Analytics Engine
Executes compression/decompression, benchmarks latency, computes SI-SDR / SNR / RTF metrics,
and produces real-time per-frame analytics.
"""

from typing import Any, Dict, Optional, Tuple, Union
import math
import os
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

from neurocodec import NeuroCodec


def calculate_snr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Computes Signal-to-Noise Ratio in dB."""
    min_len = min(len(reference), len(estimated))
    ref = reference[:min_len]
    est = estimated[:min_len]
    noise = ref - est
    signal_power = np.mean(ref**2) + 1e-9
    noise_power = np.mean(noise**2) + 1e-9
    return float(10 * np.log10(signal_power / noise_power))


def calculate_si_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    min_len = min(len(reference), len(estimated))
    ref = reference[:min_len]
    est = estimated[:min_len]

    ref_zero_mean = ref - np.mean(ref)
    est_zero_mean = est - np.mean(est)

    dot = np.dot(est_zero_mean, ref_zero_mean)
    ref_norm_sq = np.sum(ref_zero_mean**2) + 1e-9
    s_target = (dot / ref_norm_sq) * ref_zero_mean
    e_noise = est_zero_mean - s_target

    target_pwr = np.sum(s_target**2) + 1e-9
    noise_pwr = np.sum(e_noise**2) + 1e-9
    return float(10 * np.log10(target_pwr / noise_pwr))


class NeuroCodecInference:
    """
    High-performance inference wrapper with quality & real-time analytics.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: str = "config.yaml",
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # Load config
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        audio_cfg = self.config.get("audio", {})
        model_cfg = self.config.get("model", {})
        self.sample_rate = audio_cfg.get("sample_rate", 16000)

        # Instantiate model
        self.model = NeuroCodec(
            sample_rate=self.sample_rate,
            latent_dim=model_cfg.get("encoder", {}).get("latent_dim", 512),
            hidden_dim=model_cfg.get("encoder", {}).get("hidden_dim", 64),
            num_quantizers=model_cfg.get("rvq", {}).get("num_quantizers", 8),
            codebook_size=model_cfg.get("rvq", {}).get("codebook_size", 1024),
            encoder_strides=tuple(model_cfg.get("encoder", {}).get("strides", [2, 4, 5, 8])),
            decoder_strides=tuple(model_cfg.get("decoder", {}).get("strides", [8, 5, 4, 2])),
        ).to(self.device)

        # Load weights if checkpoint exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded NeuroCodec weights from {checkpoint_path}")
        else:
            print("Running NeuroCodec with initialized weights.")

        self.model.eval()

    def load_audio(self, audio_input: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Loads and prepares audio as a (1, 1, T) float32 tensor @ 16 kHz.
        """
        if isinstance(audio_input, str):
            data, sr = sf.read(audio_input, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
            if sr != self.sample_rate:
                target_len = int(waveform.shape[-1] * self.sample_rate / sr)
                waveform = F.interpolate(waveform, size=target_len, mode="linear", align_corners=False)
        elif isinstance(audio_input, np.ndarray):
            data = audio_input.astype(np.float32)
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            elif data.ndim == 2:
                waveform = torch.from_numpy(data).mean(dim=0, keepdim=True).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data)
        elif isinstance(audio_input, torch.Tensor):
            waveform = audio_input.float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        return waveform.to(self.device)

    @torch.no_grad()
    def process_audio(
        self,
        audio_input: Union[str, np.ndarray, torch.Tensor],
        bitrate_scale: float = 1.0,
        manual_quantizers: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compresses and decompresses speech with end-to-end latency and quality metrics.
        """
        x = self.load_audio(audio_input)
        duration_sec = x.shape[-1] / self.sample_rate

        # 1. Benchmark Encode
        t0 = time.perf_counter()
        enc_out = self.model.encode(
            x, manual_quantizers=manual_quantizers, bitrate_scale=bitrate_scale
        )
        t_enc = (time.perf_counter() - t0) * 1000.0  # ms

        # 2. Benchmark Decode
        t1 = time.perf_counter()
        reconstructed = self.model.decode(
            indices=enc_out["indices"],
            active_counts=enc_out["active_counts"],
            target_length=x.shape[-1],
        )
        t_dec = (time.perf_counter() - t1) * 1000.0  # ms

        total_latency_ms = t_enc + t_dec
        rtf = (total_latency_ms / 1000.0) / max(duration_sec, 1e-4)

        # 3. Quality Metrics
        ref_np = x.squeeze().cpu().numpy()
        rec_np = reconstructed.squeeze().cpu().numpy()

        snr = calculate_snr(ref_np, rec_np)
        si_sdr = calculate_si_sdr(ref_np, rec_np)

        # Bitrate metrics
        avg_kbps = enc_out["avg_bitrate_kbps"]
        raw_pcm_kbps = (16 * self.sample_rate) / 1000.0  # 256 kbps
        compression_ratio = raw_pcm_kbps / max(avg_kbps, 0.1)

        # Save to disk if requested
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            sf.write(output_path, rec_np, self.sample_rate)

        return {
            "reconstructed_waveform": rec_np,
            "original_waveform": ref_np,
            "sample_rate": self.sample_rate,
            "duration_sec": duration_sec,
            "indices": enc_out["indices"].cpu(),
            "active_counts": enc_out["active_counts"].squeeze(0).cpu().numpy(),
            "importance_scores": enc_out["importance_scores"].squeeze(0).cpu().numpy(),
            "per_frame_kbps": enc_out["per_frame_kbps"].squeeze(0).cpu().numpy(),
            "avg_bitrate_kbps": avg_kbps,
            "compression_ratio": compression_ratio,
            "encoder_latency_ms": t_enc,
            "decoder_latency_ms": t_dec,
            "total_latency_ms": total_latency_ms,
            "rtf": rtf,
            "snr_db": snr,
            "si_sdr_db": si_sdr,
        }
