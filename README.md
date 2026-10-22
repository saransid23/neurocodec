# 🎙️ NeuroCodec: Neural Audio Codec for High-Fidelity Speech Compression

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io)
[![ONNX](https://img.shields.io/badge/ONNX-Edge%20Optimized-005CED.svg?style=flat&logo=onnx)](https://onnx.ai)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**NeuroCodec** is an end-to-end, high-fidelity neural audio codec engineered specifically for 16 kHz mono speech. It achieves extreme compression ratios (**>100x** reduction relative to 256 kbps 16-bit PCM) while preserving pristine phonetic clarity, prosody, and speaker identity through **AI-driven dynamic bit allocation**, **Residual Vector Quantization (RVQ)**, **Semantic Loss**, and **Noise-Aware Suppression**.

---

## 🌟 Key Architecture & Capabilities

```
 Raw Waveform (16 kHz) [B, 1, T]
            │
            ▼
┌─────────────────────────┐
│  Convolutional Encoder  │  (Downsampling 320x -> 50 Hz frame rate)
└───────────┬─────────────┘
            ▼
     Continuous Latent z [B, 512, T_frames]
            ├──────────────────────────────────────────┐
            ▼                                          ▼
┌───────────────────────────┐             ┌─────────────────────────┐
│     Importance Network    │             │   8-Stage RVQ (512-dim) │
│ (Dynamic Bit Allocation)  │             │   1024 Codebooks/Stage  │
└───────────┬───────────────┘             └────────────┬────────────┘
            │ Quantizer Mask [B, 8, T]                 │
            └───────────────────►◄─────────────────────┘
                                  ▼
                     Quantized Latent z_q [B, 512, T]
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  Convolutional Decoder  │  (Upsampling 320x)
                     └────────────┬────────────┘
                                  ▼
                   Reconstructed Waveform [B, 1, T]
```

### 1. Core Codec Architecture
- **Sampling Rate:** 16,000 Hz Mono speech.
- **Hop Ratio:** 320x (`strides = [2, 4, 5, 8]`), producing a 50 Hz latent frame rate (20 ms frame window).
- **Residual Vector Quantizer (RVQ):** 8 codebook stages, each with 1,024 vectors of dimension 512 (10 bits per stage).
- **Bitrate Range:** Dynamically scales between **0.5 kbps** (1 stage active) and **4.0 kbps** (8 stages active), averaging ~**2.3 – 3.2 kbps**.

### 2. AI-Driven Adaptive Bitrate (Importance Network)
- Uses temporal convolutions over latent vectors to estimate per-frame acoustic and phonetic complexity.
- Dynamically allocates fewer quantizer stages to pauses, silence, and steady backgrounds while dedicating full 8-layer precision to complex transients, fricatives, and voiced vowels.

### 3. Noise-Aware Compression
- Trained with clean-noisy speech pairs mixed dynamically between 5 dB and 25 dB SNR (pink, babble, white noise).
- The encoder implicitly suppresses background acoustic interference, outputting noise-attenuated speech upon decoding.

### 4. Semantic Compression Loss
- Uses pre-trained deep acoustic representations (Wav2Vec2-BERT / multi-scale acoustic feature extractors) to penalize phonetic drift and maintain speaker timbre and prosody.

### 5. Edge AI Optimization (< 100 ms CPU Latency)
- Exportable to ONNX format with graph optimizations and dynamic batch/sequence axes.
- Benchmarked on edge CPU via ONNX Runtime:
  - **Encoder + Importance:** ~**63.97 ms** (Target: < 100 ms) ✅ **PASSED**
  - **Decoder:** ~**68.37 ms** (Target: < 100 ms) ✅ **PASSED**
  - **Real-Time Factor (RTF):** ~**0.135x** (>7x faster than real-time audio playback!)

### 6. Explainable AI Dashboard
- Interactive Streamlit web interface with real-time spectrogram comparison, active RVQ stage heatmaps, instantaneous bitrate curves, noise injection toggles, and A/B audio player.

---

## 📁 Repository Structure

```
c:\nuerocodex\
├── config.yaml              # Default hyperparameter & model configuration
├── requirements.txt         # Package dependencies
├── encoder.py               # Multi-scale dilated residual convolutional encoder
├── decoder.py               # Transposed convolutional upsampling decoder
├── quantization.py          # 8-stage RVQ with straight-through estimator (STE)
├── discriminators.py        # Multi-Period (MPD) and Multi-Scale (MSD) discriminators
├── neurocodec.py            # Unified NeuroCodec system & Importance Network
├── losses.py                # MS-STFT, Adversarial, Feature Matching, Semantic, & Bitrate losses
├── dataset.py               # Noise-aware LibriSpeech loader & acoustic noise synthesizer
├── trainer.py               # End-to-end AdamW training loop with gradient accumulation
├── train.py                 # CLI training execution script
├── inference.py             # Inference engine, RTF, SI-SDR, SNR, & real-time analytics
├── convert_to_onnx.py       # ONNX exporter & CPU latency benchmarking
├── dashboard.py             # Streamlit Explainable AI Web Dashboard
├── scripts/
│   ├── run_dashboard.bat    # 1-click dashboard launch script
│   ├── run_train.bat        # 1-click training launch script
│   └── export_onnx.bat      # 1-click ONNX export & benchmark script
└── onnx_models/             # Generated ONNX models (.onnx)
```

---

## 🚀 Quickstart Guide

### 1. Installation

Ensure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard

To test audio compression, inspect bit allocation heatmaps, and compare spectrograms:

```bash
python -m streamlit run dashboard.py
# or on Windows:
scripts\run_dashboard.bat
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Inference & Real-Time Analytics

Compress and reconstruct an audio file directly from Python:

```python
from inference import NeuroCodecInference

engine = NeuroCodecInference(checkpoint_path="checkpoints/neurocodec_latest.pt")
results = engine.process_audio(
    audio_input="path/to/speech.wav",
    output_path="reconstructed.wav",
    bitrate_scale=1.0,
)

print(f"Average Bitrate:     {results['avg_bitrate_kbps']:.2f} kbps")
print(f"Compression Ratio:   {results['compression_ratio']:.1f}x")
print(f"Total CPU Latency:   {results['total_latency_ms']:.2f} ms")
print(f"Real-Time Factor:    {results['rtf']:.3f}x")
print(f"SI-SDR Quality:      {results['si_sdr_db']:.2f} dB")
```

### 4. Edge AI Export & ONNX CPU Benchmarking

Export Encoder, Decoder, and End-to-End models to ONNX and run CPU latency benchmarks:

```bash
python convert_to_onnx.py --config config.yaml --output_dir onnx_models
```

Benchmark output:
```
================ ONNX CPU BENCHMARK SUMMARY ================
Encoder + Importance Latency: 63.97 ms  (Target: < 100 ms) -> PASSED
Decoder Latency:              68.37 ms  (Target: < 100 ms) -> PASSED
End-to-End Pipeline Latency:  134.67 ms  (RTF: 0.135x)
============================================================
```

### 5. Training on LibriSpeech

To train on your LibriSpeech train-clean-100 dataset:

```bash
python train.py --config config.yaml --data_dir /path/to/LibriSpeech/train-clean-100 --epochs 50 --batch_size 4
```

> **Note:** If `--data_dir` is omitted, NeuroCodec automatically activates its built-in realistic harmonic speech formant synthesizer, allowing zero-setup turnkey training tests.

Monitor training metrics in real-time with TensorBoard:

```bash
tensorboard --logdir runs/neurocodec
```

---

## 📐 Loss Formulation Details

$$\mathcal{L}_{\text{total}} = \lambda_{\text{stft}} \mathcal{L}_{\text{MS-STFT}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{feat}} \mathcal{L}_{\text{feat}} + \lambda_{\text{sem}} \mathcal{L}_{\text{semantic}} + \lambda_{\text{rvq}} \mathcal{L}_{\text{RVQ}} + \lambda_{\text{bitrate}} \mathcal{L}_{\text{bitrate}}$$

1. **Multi-Scale STFT ($\mathcal{L}_{\text{MS-STFT}}$):** Spectral convergence + Log-magnitude STFT loss across FFT sizes 2048, 1024, and 512.
2. **Adversarial & Feature Matching ($\mathcal{L}_{\text{adv}}, \mathcal{L}_{\text{feat}}$):** Multi-Period Discriminators ($p \in \{2, 3, 5, 7, 11\}$) and Multi-Scale Discriminators (1x, 2x, 4x pooling).
3. **Semantic Loss ($\mathcal{L}_{\text{semantic}}$):** Cosine distance and L2 distance across hidden feature representations to preserve phonetic identities.
4. **Bitrate Regularization ($\mathcal{L}_{\text{bitrate}}$):** Drives average frame quantizer activation toward a target bit allocation.
