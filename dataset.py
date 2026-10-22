"""
NeuroCodec: Dataset & Noise-Aware Speech Data Pipeline
Loads 16 kHz audio (LibriSpeech train-clean-100 or local directory), slices 1-sec segments,
and synthetically/dynamically injects noise at varying SNRs (5-25 dB) to create
noisy-clean speech pairs for noise-aware compression.
"""

from typing import List, Optional, Tuple, Union
import glob
import math
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np


class NoiseSimulator:
    """
    Real-time acoustic noise synthesizer for speech enhancement training.
    Generates white noise, pink (1/f) noise, simulated multi-speaker babble,
    and reverberant flutter.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def generate_white_noise(self, length: int) -> torch.Tensor:
        return torch.randn(1, length)

    def generate_pink_noise(self, length: int) -> torch.Tensor:
        """Generates 1/f pink noise using Voss-McCartney filtering approximation."""
        white = torch.randn(length).numpy()
        # Simple IIR filter for -3dB/octave slope
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        from scipy.signal import lfilter
        pink = lfilter(b, a, white)
        pink = pink / (np.max(np.abs(pink)) + 1e-8)
        return torch.from_numpy(pink).float().unsqueeze(0)

    def generate_babble_noise(self, length: int) -> torch.Tensor:
        """Simulates overlapping chatter of several distant speakers using formant modulation."""
        t = torch.linspace(0, length / self.sample_rate, length)
        noise = torch.zeros(1, length)
        # Combine random modulating frequencies representing vocal tracts
        for _ in range(5):
            f0 = random.uniform(100, 300)
            f1 = random.uniform(500, 1500)
            f2 = random.uniform(1800, 2800)
            mod = torch.sin(2 * math.pi * f0 * t) * 0.5 + torch.sin(2 * math.pi * f1 * t) * 0.3
            envelope = (torch.sin(2 * math.pi * random.uniform(1, 4) * t) ** 2)
            noise = noise + (mod * envelope).unsqueeze(0)
        noise = noise / (noise.abs().max() + 1e-8)
        return noise

    def sample_noise(self, length: int) -> torch.Tensor:
        choice = random.choice(["white", "pink", "babble"])
        if choice == "white":
            return self.generate_white_noise(length)
        elif choice == "pink":
            return self.generate_pink_noise(length)
        else:
            return self.generate_babble_noise(length)

    def mix_snr(
        self, clean: torch.Tensor, noise: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """
        Mix clean speech with noise to attain exact target Signal-to-Noise Ratio (SNR) in dB.
        """
        clean_pwr = torch.mean(clean**2) + 1e-8
        noise_pwr = torch.mean(noise**2) + 1e-8

        # Target noise power for desired SNR
        target_noise_pwr = clean_pwr / (10 ** (snr_db / 10.0))
        scaled_noise = noise * torch.sqrt(target_noise_pwr / noise_pwr)

        return clean + scaled_noise


class SpeechDataset(Dataset):
    """
    Noise-Aware Speech Dataset.
    Loads audio clips, segments them into 1-second chunks (16000 samples @ 16 kHz),
    and creates (noisy_input, clean_target) pairs for noise-robust compression training.
    """

    def __init__(
        self,
        audio_dir: Optional[str] = None,
        sample_rate: int = 16000,
        segment_length: int = 16000,
        noise_prob: float = 0.7,
        snr_range: Tuple[float, float] = (5.0, 25.0),
        num_synthetic_samples: int = 200,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.noise_prob = noise_prob
        self.snr_range = snr_range
        self.noise_sim = NoiseSimulator(sample_rate=sample_rate)

        # Collect audio files
        self.file_paths: List[str] = []
        if audio_dir and os.path.isdir(audio_dir):
            for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
                self.file_paths.extend(glob.glob(os.path.join(audio_dir, "**", ext), recursive=True))

        self.num_synthetic_samples = num_synthetic_samples
        self.use_synthetic = len(self.file_paths) == 0

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.num_synthetic_samples
        return len(self.file_paths)

    def _generate_synthetic_speech(self) -> torch.Tensor:
        """
        Synthesizes realistic vowel/phoneme harmonic signals with pitch contour and envelopes.
        Used as immediate turnkey fallback when dataset is not yet pre-downloaded.
        """
        length = self.segment_length
        t = torch.linspace(0, length / self.sample_rate, length)
        # Base pitch between 100 Hz and 250 Hz with subtle vibrato
        f0 = random.uniform(110, 220)
        vibrato = torch.sin(2 * math.pi * 5.0 * t) * 5.0
        instantaneous_phase = 2 * math.pi * torch.cumsum((f0 + vibrato) / self.sample_rate, dim=0)

        # Harmonics
        signal = torch.zeros(length)
        for h in range(1, 8):
            amp = (1.0 / h) * random.uniform(0.7, 1.0)
            signal += amp * torch.sin(instantaneous_phase * h)

        # Formant envelope filtering
        envelope = torch.sin(math.pi * t / (length / self.sample_rate)) ** 2
        speech = signal * envelope

        # Normalize
        speech = speech / (torch.max(torch.abs(speech)) + 1e-6)
        return speech.unsqueeze(0)

    def _load_audio_file(self, path: str) -> torch.Tensor:
        """Loads and converts file to mono 16 kHz float32 tensor."""
        data, sr = sf.read(path, dtype="float32")
        # To mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        waveform = torch.from_numpy(data).float().unsqueeze(0)

        # Resample if needed
        if sr != self.sample_rate:
            # Linear / polynomial interpolation resample
            orig_len = waveform.shape[-1]
            target_len = int(orig_len * self.sample_rate / sr)
            waveform = F.interpolate(
                waveform.unsqueeze(0), size=target_len, mode="linear", align_corners=False
            ).squeeze(0)

        # Segment or pad to self.segment_length
        curr_len = waveform.shape[-1]
        if curr_len < self.segment_length:
            pad = self.segment_length - curr_len
            waveform = F.pad(waveform, (0, pad), mode="reflect")
        elif curr_len > self.segment_length:
            # Random crop
            start = random.randint(0, curr_len - self.segment_length)
            waveform = waveform[:, start : start + self.segment_length]

        # Peak normalization
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-4:
            waveform = waveform / max_val * 0.95

        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            noisy_audio: (1, 16000)
            clean_audio: (1, 16000)
        """
        if self.use_synthetic:
            clean = self._generate_synthetic_speech()
        else:
            clean = self._load_audio_file(self.file_paths[idx])

        # Apply noise augmentation with probability self.noise_prob
        if random.random() < self.noise_prob:
            snr = random.uniform(*self.snr_range)
            noise = self.noise_sim.sample_noise(self.segment_length)
            noisy = self.noise_sim.mix_snr(clean, noise, snr)
            # Clip to valid amplitude range
            noisy = torch.clamp(noisy, -1.0, 1.0)
        else:
            noisy = clean.clone()

        return noisy, clean
