import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

from unet import UNet
from preprocess.stft_utils import wav_to_mag_phase, mag_phase_to_wav
from metrics import calculate_metrics   # must take numpy arrays

# ===============================
# CONFIG
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000

model_path = "models/unet_final_continued.pt"
clean_file = "test_audio/clean.wav"
noisy_file = "test_audio/noisy.wav"


# ===============================
# LOAD MODEL
# ===============================
model = UNet().to(DEVICE)
checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ===============================
# LOAD AUDIO
# ===============================
clean, sr = torchaudio.load(clean_file)
noisy, sr = torchaudio.load(noisy_file)

clean = clean.to(DEVICE)
noisy = noisy.to(DEVICE)


# ===============================
# ENHANCEMENT
# ===============================
mag, phase = wav_to_mag_phase(noisy)
mag = mag.unsqueeze(1).to(DEVICE)
phase = phase.to(DEVICE)

with torch.no_grad():
    enhanced_mag = model(mag)

enhanced_mag = enhanced_mag.squeeze(1)
enhanced = mag_phase_to_wav(enhanced_mag, phase)


# ===============================
# CROP ALL SIGNALS
# ===============================
min_len = min(clean.shape[-1], noisy.shape[-1], enhanced.shape[-1])
clean = clean[:, :min_len]
noisy = noisy[:, :min_len]
enhanced = enhanced[:, :min_len]

clean_np = clean.squeeze().cpu().numpy()
noisy_np = noisy.squeeze().cpu().numpy()
enhanced_np = enhanced.squeeze().cpu().numpy()


# ===============================
# METRICS
# ===============================
stoi_score, pesq_score, snr_score = calculate_metrics(clean_np, enhanced_np)

print("\n=========== METRICS ===========")
print(f"STOI : {stoi_score:.4f}")
print(f"PESQ : {pesq_score:.4f}")
print(f"SNR  : {snr_score:.2f} dB")
print("================================\n")


# ===============================
# GRAPH FUNCTIONS
# ===============================

def plot_waveform(clean_np, noisy_np, enhanced_np):
    plt.figure(figsize=(12, 5))
    plt.plot(clean_np, label="Clean")
    plt.plot(noisy_np, label="Noisy")
    plt.plot(enhanced_np, label="Enhanced")
    plt.title("Waveform Comparison",fontsize=20)
    plt.xlabel("Time Samples",fontsize=20)
    plt.ylabel("Amplitude",fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spectrogram_with_axes(signal, sr, title):
    f, t, Zxx = stft(signal, sr, nperseg=512, noverlap=256)
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-8)

    plt.figure(figsize=(10, 4))
    plt.imshow(
        magnitude_db,
        aspect='auto',
        origin='lower',
        extent=[t.min(), t.max(), f.min(), f.max()],
        cmap='viridis'
    )
    plt.colorbar(label='Magnitude (dB)')
    plt.title(title,fontsize=16)
    plt.xlabel("Time (seconds)",fontsize=20)
    plt.ylabel("Frequency (Hz)",fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_difference_spectrogram_with_axes(noisy_np, enhanced_np, sr=16000):
    f_n, t_n, S_noisy = stft(noisy_np, sr, nperseg=512, noverlap=256)
    f_e, t_e, S_enh = stft(enhanced_np, sr, nperseg=512, noverlap=256)

    diff_db = (
        20 * np.log10(np.abs(S_enh) + 1e-8)
        - 20 * np.log10(np.abs(S_noisy) + 1e-8)
    )

    plt.figure(figsize=(10, 4))
    plt.imshow(
        diff_db,
        aspect="auto",
        origin="lower",
        extent=[t_n.min(), t_n.max(), f_n.min(), f_n.max()],
        cmap="coolwarm"
    )
    plt.colorbar(label="Difference (dB)")
    plt.title("Difference Spectrogram (Enhanced - Noisy)",fontsize=20)
    plt.xlabel("Time (seconds)",fontsize=20)
    plt.ylabel("Frequency (Hz)",fontsize=20)
    plt.tight_layout()
    plt.show()


# ===============================
# PLOTS
# ===============================

plot_waveform(clean_np, noisy_np, enhanced_np)

plot_spectrogram_with_axes(clean_np, SR, "Clean Spectrogram")
plot_spectrogram_with_axes(noisy_np, SR, "Noisy Spectrogram")
plot_spectrogram_with_axes(enhanced_np, SR, "Enhanced Spectrogram")

plot_difference_spectrogram_with_axes(noisy_np, enhanced_np, SR)
