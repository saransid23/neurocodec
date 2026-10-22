"""
Create high-quality sample speech and noisy audio files for NeuroCodec testing.
"""

import os
import math
import numpy as np
import soundfile as sf
import scipy.signal as signal

os.makedirs("samples", exist_ok=True)
sr = 16000

def synthesize_speech(
    f0_base: float = 130.0,
    duration: float = 3.5,
    formants_sequence=None,
    vibrato_rate: float = 4.5,
    vibrato_depth: float = 4.0,
    pauses=True,
):
    """
    Synthesizes speech-like phonetic utterances using source-filter model:
    Glottal pulse harmonic train filtered by dynamic vocal tract formants (F1, F2, F3).
    """
    total_samples = int(sr * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    
    # 1. Pitch contour with natural prosody and vibrato
    # Intonation curve: rises, fluctuates, and falls
    intonation = 1.0 + 0.15 * np.sin(2 * np.pi * 0.8 * t) - 0.1 * (t / duration)
    pitch_contour = f0_base * intonation + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    phase = 2 * np.pi * np.cumsum(pitch_contour / sr)
    
    # Glottal source (harmonic series with ~ -12dB/octave rolloff)
    source = np.zeros(total_samples)
    for h in range(1, 25):
        amp = (1.0 / (h ** 1.1)) * (0.8 + 0.4 * np.sin(h * 0.5))
        source += amp * np.sin(phase * h)
    
    # Glottal closure pulse shaping
    source = np.tanh(source * 1.5)
    
    # 2. Formant trajectories (simulating transitions between vowels /a/, /i/, /u/, /o/)
    if formants_sequence is None:
        # Timed vowel transitions: /a/ (750, 1200) -> /i/ (280, 2250) -> /u/ (320, 800) -> /o/ (500, 1000)
        formants_sequence = [
            (0.0, 750, 1250, 2600),   # /a/
            (0.8, 280, 2250, 2900),   # /i/
            (1.6, 330, 850, 2400),    # /u/
            (2.4, 500, 1050, 2500),   # /o/
            (3.2, 650, 1150, 2550),   # /a/
        ]
    
    # Interpolate formant center frequencies across time
    f1_t = np.interp(t, [p[0] for p in formants_sequence], [p[1] for p in formants_sequence])
    f2_t = np.interp(t, [p[0] for p in formants_sequence], [p[2] for p in formants_sequence])
    f3_t = np.interp(t, [p[0] for p in formants_sequence], [p[3] for p in formants_sequence])
    
    # Resonator filters (cascade of bandpass IIR formants)
    filtered = np.zeros(total_samples)
    
    # Chunked processing to allow time-varying formant filtering
    hop = 160  # 10 ms frames
    for i in range(0, total_samples, hop):
        end = min(i + hop, total_samples)
        mid_idx = (i + end) // 2
        f1, f2, f3 = f1_t[mid_idx], f2_t[mid_idx], f3_t[mid_idx]
        
        # Design 2nd order resonators for each formant
        for fc, bw, gain in [(f1, 80, 1.0), (f2, 110, 0.6), (f3, 140, 0.3)]:
            w0 = fc / (sr / 2)
            q = fc / bw
            if 0 < w0 < 0.95:
                b, a = signal.iirpeak(w0, q)
                chunk = signal.lfilter(b, a, source[i:end])
                filtered[i:end] += gain * chunk
    
    # 3. Add unvoiced fricative consonants at syllable onsets (/s/, /sh/)
    fricative_noise = np.random.randn(total_samples)
    b_fric, a_fric = signal.butter(4, [3500 / (sr / 2), 7000 / (sr / 2)], btype="bandpass")
    fricative = signal.lfilter(b_fric, a_fric, fricative_noise)
    
    # Fricative bursts at specific timestamps (e.g. at 0.6s and 2.1s)
    burst_mask = np.zeros(total_samples)
    for onset in [0.65, 2.15]:
        start_samp = int(onset * sr)
        burst_len = int(0.08 * sr)
        if start_samp + burst_len < total_samples:
            env = np.hanning(burst_len)
            burst_mask[start_samp : start_samp + burst_len] = env * 0.4
    
    speech = filtered + fricative * burst_mask
    
    # 4. Syllable amplitude envelopes and conversational pauses
    # Creates natural speech cadence with a 0.5s pause in the middle to test dynamic bit allocation
    syllable_rate = 3.5  # syllables per sec
    amplitude_env = 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t))
    
    if pauses:
        # Silence/pause between 1.3s and 1.8s
        pause_start = int(1.3 * sr)
        pause_end = int(1.8 * sr)
        pause_curve = np.ones(total_samples)
        fade_len = int(0.05 * sr)
        pause_curve[pause_start - fade_len : pause_start] = np.linspace(1, 0, fade_len)
        pause_curve[pause_start : pause_end] = 0.005  # Near silence
        pause_curve[pause_end : pause_end + fade_len] = np.linspace(0, 1, fade_len)
        amplitude_env *= pause_curve
    
    speech = speech * amplitude_env
    # Normalize peak amplitude to -1 dB
    speech = speech / (np.max(np.abs(speech)) + 1e-6) * 0.89
    return speech.astype(np.float32)

# 1. Generate Clean Male Speech
print("Generating clean male speech sample...")
male_speech = synthesize_speech(f0_base=125.0, duration=3.5, vibrato_rate=4.2)
sf.write("samples/speech_male_clean.wav", male_speech, sr)

# 2. Generate Clean Female Speech
print("Generating clean female speech sample...")
female_speech = synthesize_speech(f0_base=215.0, duration=3.5, vibrato_rate=5.0)
sf.write("samples/speech_female_clean.wav", female_speech, sr)

# 3. Generate Babble Noise and Mix (12 dB SNR)
print("Generating noisy speech with babble noise (12 dB SNR)...")
t = np.linspace(0, len(male_speech) / sr, len(male_speech), endpoint=False)
babble = np.zeros_like(male_speech)
for f_mod in [160, 220, 280, 340, 420]:
    b_env = np.sin(2 * np.pi * np.random.uniform(2, 6) * t) ** 2
    babble += np.sin(2 * np.pi * f_mod * t) * b_env * np.random.randn(len(t))
babble = babble / (np.max(np.abs(babble)) + 1e-6)

# Mix at 12 dB SNR
pwr_clean = np.mean(male_speech ** 2)
pwr_babble = np.mean(babble ** 2)
target_babble_pwr = pwr_clean / (10 ** (12.0 / 10.0))
noisy_babble = male_speech + babble * np.sqrt(target_babble_pwr / (pwr_babble + 1e-8))
noisy_babble = np.clip(noisy_babble, -0.95, 0.95)
sf.write("samples/speech_noisy_babble_12db.wav", noisy_babble, sr)

# 4. Generate Environmental Pink Noise (8 dB SNR)
print("Generating noisy speech with environmental pink noise (8 dB SNR)...")
white = np.random.randn(len(female_speech))
b_pink, a_pink = [0.049922, -0.09599, 0.05061, -0.0044], [1.0, -2.4949, 2.0172, -0.522]
pink = signal.lfilter(b_pink, a_pink, white)
pink = pink / (np.max(np.abs(pink)) + 1e-6)

pwr_clean_f = np.mean(female_speech ** 2)
pwr_pink = np.mean(pink ** 2)
target_pink_pwr = pwr_clean_f / (10 ** (8.0 / 10.0))
noisy_pink = female_speech + pink * np.sqrt(target_pink_pwr / (pwr_pink + 1e-8))
noisy_pink = np.clip(noisy_pink, -0.95, 0.95)
sf.write("samples/speech_noisy_pink_8db.wav", noisy_pink, sr)

print("Generated sample files in c:\\nuerocodex\\samples:")
for f in os.listdir("samples"):
    path = os.path.join("samples", f)
    print(f"  - {f} ({os.path.getsize(path)} bytes)")
