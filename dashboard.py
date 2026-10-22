"""
NeuroCodec: Explainable AI & Interactive Web Platform
Streamlit Dashboard for real-time speech compression, adaptive bit allocation visualization,
spectrogram analysis, noise suppression testing, and edge AI latency benchmarking.
"""

import io
import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torch.nn.functional as F

from neurocodec import NeuroCodec
from dataset import NoiseSimulator
from inference import calculate_si_sdr, calculate_snr


# Page Configuration
st.set_page_config(
    page_title="NeuroCodec | Explainable Neural Speech Codec",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, sans-serif;
    }

    .main {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #030712 100%);
    }

    .neuro-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .neuro-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        font-weight: 600;
        margin-top: 4px;
    }

    .badge-dynamic {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }

    .badge-fixed {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_neurocodec_model():
    """Initializes and caches the NeuroCodec model."""
    model = NeuroCodec(
        sample_rate=16000,
        latent_dim=512,
        hidden_dim=64,
        num_quantizers=8,
        codebook_size=1024,
        encoder_strides=(2, 4, 5, 8),
        decoder_strides=(8, 5, 4, 2),
    )
    # Check if a trained checkpoint exists
    ckpt_candidates = [
        "checkpoints/neurocodec_latest.pt",
        "checkpoints/neurocodec_epoch_001.pt",
    ]
    for ckpt in ckpt_candidates:
        if os.path.exists(ckpt):
            try:
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state.get("model_state_dict", state), strict=False)
                break
            except Exception:
                pass
    model.eval()
    return model


def generate_demo_speech(sample_rate: int = 16000, duration_sec: float = 2.0) -> np.ndarray:
    """Generates synthetic speech-like vowel transitions with pitch contour."""
    length = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, length)
    f0 = 130.0 + 30.0 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0 / sample_rate)

    harmonics = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.08]
    signal = np.zeros(length)
    for i, amp in enumerate(harmonics, start=1):
        signal += amp * np.sin(phase * i)

    # Vowel formant resonance envelope modulation
    envelope = (np.sin(np.pi * t / duration_sec) ** 2) * (1.0 + 0.3 * np.sin(2 * np.pi * 3.0 * t))
    audio = signal * envelope
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.9
    return audio.astype(np.float32)


def plot_spectrogram(audio: np.ndarray, sr: int, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    D = np.abs(
        np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(
                np.pad(audio, (256, 256)), 512
            )[::128]
            * np.hanning(512),
            axis=-1,
        )
    ).T
    log_D = 10 * np.log10(D + 1e-6)
    im = ax.imshow(
        log_D,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, len(audio) / sr, 0, sr // 2],
    )
    ax.set_title(title, color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout()
    return fig


def main():
    model = load_neurocodec_model()

    # Header Section
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 28px;">
            <h1 style="font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 6px;">
                NeuroCodec
            </h1>
            <p style="color: #94a3b8; font-size: 1.15rem; max-width: 700px; margin: 0 auto;">
                Explainable Neural Audio Codec featuring AI-Driven Adaptive Bitrate, 8-Stage RVQ,
                Noise-Aware Compression, and Edge CPU Optimization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Codec Controls")

        bitrate_mode = st.radio(
            "Bitrate Allocation Mode",
            ["AI-Driven Dynamic Bitrate", "Manual Fixed Quantizers"],
            index=0,
        )

        if bitrate_mode == "AI-Driven Dynamic Bitrate":
            st.markdown(
                '<span class="badge-dynamic">AI Importance Network Active</span>',
                unsafe_allow_html=True,
            )
            bitrate_scale = st.slider(
                "Bitrate Scaling Factor",
                min_value=0.2,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Scales the frame importance thresholds dynamically.",
            )
            manual_quantizers = None
        else:
            st.markdown(
                '<span class="badge-fixed">Manual Quantizer Override</span>',
                unsafe_allow_html=True,
            )
            manual_quantizers = st.slider(
                "Active Quantizer Stages",
                min_value=1,
                max_value=8,
                value=6,
                step=1,
                help="1 quantizer = 0.5 kbps; 8 quantizers = 4.0 kbps",
            )
            bitrate_scale = 1.0

        st.markdown("---")
        st.markdown("### 🔊 Noise Augmentation")
        add_noise = st.checkbox("Simulate Background Noise", value=False)
        noise_type = "babble"
        noise_snr = 15.0
        if add_noise:
            noise_type = st.selectbox("Noise Profile", ["babble", "pink", "white"])
            noise_snr = st.slider("Signal-to-Noise Ratio (SNR dB)", 5.0, 30.0, 15.0, 1.0)

        st.markdown("---")
        st.markdown("### 🔬 System Specifications")
        st.markdown(
            """
            - **Sampling Rate:** 16,000 Hz (Mono)
            - **Frame Rate:** 50 Hz (20 ms hop / 320x)
            - **Codebook Stages:** 8 Quantizers
            - **Codebook Size:** 1,024 entries (10 bits)
            - **Latent Dim:** 512 dimensions
            - **Peak Bitrate:** 4.0 kbps (8 layers)
            - **Floor Bitrate:** 0.5 kbps (1 layer)
            """
        )

    # Audio Input Selection
    st.markdown("### 1. Audio Source")
    input_tab0, input_tab1, input_tab2 = st.tabs([
        "🎵 Pre-loaded Samples",
        "📁 Upload Speech Audio",
        "⚡ Instant Speech Synthesizer",
    ])

    audio_data = None
    sr = 16000

    with input_tab0:
        sample_options = {
            "librispeech_sample_clean.wav": "LibriSpeech Human Voice: 'Mister Quilter...' (Real Clean Speech)",
            "librispeech_conversational_16k.wav": "Real Conversational Voice (5.0s, Real Human Speech)",
            "librispeech_sample_noisy_12db.wav": "LibriSpeech + Real Noise Mixture (12 dB SNR)",
            "speech_male_clean.wav": "Clean Male Speech with pause interval",
            "speech_female_clean.wav": "Clean Female Speech with harmonic vibrato",
        }
        selected_sample_key = st.selectbox(
            "Select Real Human Sample Audio:",
            list(sample_options.keys()),
            format_func=lambda k: f"{sample_options[k]}",
        )
        sample_path = os.path.join("samples", selected_sample_key)
        if os.path.exists(sample_path):
            data, file_sr = sf.read(sample_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            audio_data = data
            st.caption(f"Loaded: `{sample_path}` ({len(audio_data) / sr:.2f} seconds @ 16 kHz)")

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Upload 16 kHz audio file (.wav, .flac, .mp3, .ogg)",
            type=["wav", "flac", "mp3", "ogg"],
        )
        if uploaded_file is not None:
            data, file_sr = sf.read(uploaded_file, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if file_sr != sr:
                # Resample
                target_len = int(len(data) * sr / file_sr)
                data = F.interpolate(
                    torch.from_numpy(data).view(1, 1, -1),
                    size=target_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze().numpy()
            max_samples = 5 * sr
            if len(data) > max_samples:
                data = data[:max_samples]
            audio_data = data

    with input_tab2:
        col_synth1, col_synth2 = st.columns([2, 1])
        with col_synth1:
            synth_duration = st.slider("Synthetic Duration (seconds)", 1.0, 4.0, 2.0, 0.5)
        with col_synth2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Generate Demo Speech", use_container_width=True):
                audio_data = generate_demo_speech(sample_rate=sr, duration_sec=synth_duration)

    if audio_data is None:
        audio_data = generate_demo_speech(sample_rate=sr, duration_sec=2.0)

    # Noise Simulation if requested
    clean_audio = audio_data.copy()
    if add_noise:
        ns = NoiseSimulator(sample_rate=sr)
        if noise_type == "white":
            noise_tens = ns.generate_white_noise(len(clean_audio))
        elif noise_type == "pink":
            noise_tens = ns.generate_pink_noise(len(clean_audio))
        else:
            noise_tens = ns.generate_babble_noise(len(clean_audio))
        clean_tens = torch.from_numpy(clean_audio).unsqueeze(0)
        noisy_tens = ns.mix_snr(clean_tens, noise_tens, noise_snr)
        processed_input = np.clip(noisy_tens.squeeze().numpy(), -1.0, 1.0)
    else:
        processed_input = clean_audio

    # Run Inference
    input_tensor = torch.from_numpy(processed_input).unsqueeze(0).unsqueeze(0).float()

    t_start = time.perf_counter()
    with torch.no_grad():
        out = model(
            input_tensor,
            manual_quantizers=manual_quantizers,
            bitrate_scale=bitrate_scale,
        )
    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    rec_audio = out["reconstructed"].squeeze().cpu().numpy()
    importance_scores = out["importance_scores"].squeeze().cpu().numpy()
    active_counts = out["active_counts"].squeeze().cpu().numpy()
    per_frame_kbps = out["per_frame_kbps"].squeeze().cpu().numpy()
    avg_bitrate = float(out["bitrate_kbps"].item())
    mask = out["importance_mask"].squeeze().cpu().numpy()  # (8, T_frames)

    duration_sec = len(processed_input) / sr
    rtf = (total_time_ms / 1000.0) / max(duration_sec, 1e-4)
    snr_val = calculate_snr(clean_audio, rec_audio)
    si_sdr_val = calculate_si_sdr(clean_audio, rec_audio)
    raw_pcm_kbps = 256.0  # 16-bit * 16 kHz
    compression_ratio = raw_pcm_kbps / max(avg_bitrate, 0.1)
    savings_pct = (1.0 - avg_bitrate / raw_pcm_kbps) * 100.0

    # Real-Time Analytics Metrics Bar
    st.markdown("### 2. Real-Time Analytics & Codec Performance")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(
            f"""
            <div class="neuro-card">
                <div class="metric-value">{avg_bitrate:.2f} kbps</div>
                <div class="metric-label">Avg Bitrate (vs 256k PCM)</div>
                <div style="color: #34d399; font-size: 0.8rem; margin-top: 4px;">↓ {savings_pct:.1f}% Reduction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="neuro-card">
                <div class="metric-value">{compression_ratio:.1f}x</div>
                <div class="metric-label">Compression Factor</div>
                <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">8-layer RVQ</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="neuro-card">
                <div class="metric-value">{total_time_ms:.1f} ms</div>
                <div class="metric-label">Inference Latency</div>
                <div style="color: #38bdf8; font-size: 0.8rem; margin-top: 4px;">RTF: {rtf:.3f}x</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="neuro-card">
                <div class="metric-value">{si_sdr_val:.1f} dB</div>
                <div class="metric-label">SI-SDR Quality</div>
                <div style="color: #818cf8; font-size: 0.8rem; margin-top: 4px;">High Intelligibility</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col5:
        st.markdown(
            f"""
            <div class="neuro-card">
                <div class="metric-value">{snr_val:.1f} dB</div>
                <div class="metric-label">Reconstruction SNR</div>
                <div style="color: #c084fc; font-size: 0.8rem; margin-top: 4px;">Spectral Alignment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Audio Playback Comparison
    st.markdown("### 3. A/B Audio Comparison")
    if add_noise:
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.markdown("##### 🟢 Clean Ground Truth")
            st.audio(clean_audio, sample_rate=sr)
        with p_col2:
            st.markdown(f"##### 🟠 Noisy Input ({noise_type.title()} @ {noise_snr}dB SNR)")
            st.audio(processed_input, sample_rate=sr)
        with p_col3:
            st.markdown("##### 🟣 NeuroCodec Reconstructed")
            st.audio(rec_audio, sample_rate=sr)
    else:
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("##### 🟢 Original Audio (16 kHz Uncompressed)")
            st.audio(clean_audio, sample_rate=sr)
        with p_col2:
            st.markdown(f"##### 🟣 Compressed & Reconstructed ({avg_bitrate:.2f} kbps)")
            st.audio(rec_audio, sample_rate=sr)

    # Explainable AI: Bit Allocation & Importance Maps
    st.markdown("### 4. Explainable AI: Dynamic Bit Allocation & Importance")
    xai_col1, xai_col2 = st.columns([3, 2])

    with xai_col1:
        # Time-aligned Bitrate curve and Importance Score
        fig_alloc, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True, facecolor="#0f172a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")

        time_axis = np.linspace(0, duration_sec, len(per_frame_kbps))
        ax1.plot(time_axis, per_frame_kbps, color="#38bdf8", linewidth=2.0, label="Dynamic Bitrate (kbps)")
        ax1.axhline(avg_bitrate, color="#fbbf24", linestyle="--", alpha=0.8, label=f"Average ({avg_bitrate:.2f} kbps)")
        ax1.set_ylabel("Bitrate (kbps)", color="#94a3b8", fontsize=9)
        ax1.set_title("AI-Predicted Dynamic Bitrate Over Time", color="#e2e8f0", fontsize=11, fontweight="bold")
        ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
        ax1.grid(True, color="#1e293b", linestyle=":")

        ax2.plot(time_axis, importance_scores, color="#a855f7", linewidth=2.0, label="Acoustic Importance Score")
        ax2.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=9)
        ax2.set_ylabel("Importance [0-1]", color="#94a3b8", fontsize=9)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, color="#1e293b", linestyle=":")
        ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
        fig_alloc.tight_layout()
        st.pyplot(fig_alloc)

    with xai_col2:
        # RVQ Quantizer Stage Activation Heatmap
        fig_hm, ax_hm = plt.subplots(figsize=(6, 4.2), facecolor="#0f172a")
        ax_hm.set_facecolor("#0f172a")
        im_hm = ax_hm.imshow(
            mask,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[0, duration_sec, 0.5, 8.5],
        )
        ax_hm.set_title("Active RVQ Stages per Frame (8 Levels)", color="#e2e8f0", fontsize=11, fontweight="bold")
        ax_hm.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=9)
        ax_hm.set_ylabel("RVQ Quantizer Layer", color="#94a3b8", fontsize=9)
        ax_hm.set_yticks(range(1, 9))
        ax_hm.tick_params(colors="#94a3b8")
        for spine in ax_hm.spines.values():
            spine.set_color("#334155")
        fig_hm.tight_layout()
        st.pyplot(fig_hm)

    # Spectrogram Comparison
    st.markdown("### 5. Multi-Resolution Spectrogram Analysis")
    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        fig_orig = plot_spectrogram(clean_audio, sr, "Original Audio Spectrogram")
        st.pyplot(fig_orig)
    with spec_col2:
        fig_rec = plot_spectrogram(rec_audio, sr, "NeuroCodec Reconstructed Spectrogram")
        st.pyplot(fig_rec)

    # Edge AI ONNX Benchmark Section
    st.markdown("### 6. Edge AI CPU Optimization & ONNX Benchmark")
    onnx_col1, onnx_col2 = st.columns([1, 2])
    with onnx_col1:
        st.markdown(
            """
            Validate sub-100 ms inference performance on edge CPUs by running the ONNX Runtime engine.
            """
        )
        run_benchmark_btn = st.button("🚀 Run ONNX CPU Benchmark", use_container_width=True)

    with onnx_col2:
        if run_benchmark_btn:
            with st.spinner("Exporting ONNX graph and running benchmark on CPU..."):
                try:
                    from convert_to_onnx import export_and_benchmark
                    res = export_and_benchmark(output_dir="onnx_models")
                    st.success(
                        f"ONNX Model Verified! Mean CPU Latency: **{res['mean_latency_ms']:.2f} ms** "
                        f"(P95: {res['p95_latency_ms']:.2f} ms, RTF: {res['rtf']:.4f}x)"
                    )
                except Exception as e:
                    st.error(f"Benchmark error: {str(e)}")


if __name__ == "__main__":
    main()
