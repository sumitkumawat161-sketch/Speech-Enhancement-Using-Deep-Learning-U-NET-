import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

from unet import UNet
from preprocess.stft_utils import wav_to_mag_phase, mag_phase_to_wav
from metrics import calculate_metrics



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
test_clean_dir = "test_set/clean/"
test_noisy_dir = "test_set/noisy/"
model_path = "models/unet_final_continued.pt"



def plot_waveforms(clean_np, noisy_np, enhanced_np, title):
    plt.figure(figsize=(12, 5))
    plt.plot(clean_np, label="Clean", alpha=0.8)
    plt.plot(noisy_np, label="Noisy", alpha=0.8)
    plt.plot(enhanced_np, label="Enhanced", alpha=0.8)
    plt.title(title)
    plt.xlabel("Time Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spectrogram(sig, sr, title):
    f, t, Z = stft(sig, sr)
    plt.figure(figsize=(10, 4))
    plt.imshow(20*np.log10(np.abs(Z)+1e-6), aspect="auto", origin="lower")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_average_metrics(noisy_stoi, noisy_pesq, noisy_snr,
                         enh_stoi, enh_pesq, enh_snr):
    labels = ["STOI", "PESQ", "SNR"]
    noisy_vals = [np.mean(noisy_stoi), np.mean(noisy_pesq), np.mean(noisy_snr)]
    enh_vals = [np.mean(enh_stoi), np.mean(enh_pesq), np.mean(enh_snr)]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, noisy_vals, width, label="Noisy")
    plt.bar(x + width/2, enh_vals, width, label="Enhanced")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Average STOI / PESQ / SNR Before & After Enhancement")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_per_file_improvements(noisy_list, enhanced_list, metric_name):
    diff = np.array(enhanced_list) - np.array(noisy_list)

    plt.figure(figsize=(10, 4))
    plt.plot(diff, marker="o")
    plt.title(f"{metric_name} Improvement Per File")
    plt.xlabel("File Index")
    plt.ylabel("Improvement")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histograms(noisy_list, enhanced_list, metric_name):
    diff = np.array(enhanced_list) - np.array(noisy_list)

    plt.figure(figsize=(7, 4))
    plt.hist(diff, bins=10, alpha=0.8)
    plt.title(f"{metric_name} Improvement Histogram")
    plt.xlabel("Improvement")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()



print("Loading model...")
model = UNet().to(DEVICE)
checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()



noisy_stoi_list = []
noisy_pesq_list = []
noisy_snr_list = []

enh_stoi_list = []
enh_pesq_list = []
enh_snr_list = []



def enhance(noisy_tensor):
    mag, phase = wav_to_mag_phase(noisy_tensor)
    mag = mag.unsqueeze(1).to(DEVICE)
    phase = phase.to(DEVICE)

    with torch.no_grad():
        enhanced_mag = model(mag)

    enhanced_mag = enhanced_mag.squeeze(1)
    return mag_phase_to_wav(enhanced_mag, phase)



file_list = sorted(os.listdir(test_clean_dir))
print(f"Evaluating {len(file_list)} files...\n")

for clean_fname in file_list:
    num = clean_fname.split("_")[0]
    noisy_candidates = [f for f in os.listdir(test_noisy_dir) if f.startswith(num)]
    if not noisy_candidates:
        continue
    noisy_fname = noisy_candidates[0]

    # Load audio
    clean, _ = torchaudio.load(os.path.join(test_clean_dir, clean_fname))
    noisy, _ = torchaudio.load(os.path.join(test_noisy_dir, noisy_fname))

    clean = clean.to(DEVICE)
    noisy = noisy.to(DEVICE)

    # Enhance
    enhanced = enhance(noisy)

    
    min_len = min(clean.shape[-1], noisy.shape[-1], enhanced.shape[-1])
    clean = clean[:, :min_len]
    noisy = noisy[:, :min_len]
    enhanced = enhanced[:, :min_len]

    
    clean_np = clean.squeeze().cpu().numpy()
    noisy_np = noisy.squeeze().cpu().numpy()
    enhanced_np = enhanced.squeeze().cpu().numpy()

    
    sn, pn, nn = calculate_metrics(clean_np, noisy_np)
    se, pe, ne = calculate_metrics(clean_np, enhanced_np)

    noisy_stoi_list.append(sn)
    noisy_pesq_list.append(pn)
    noisy_snr_list.append(nn)

    enh_stoi_list.append(se)
    enh_pesq_list.append(pe)
    enh_snr_list.append(ne)

    print(f"{clean_fname}:")
    print(f"   NOISY:    STOI={sn:.4f}, PESQ={pn:.4f}, SNR={nn:.2f}")
    print(f"   ENHANCED: STOI={se:.4f}, PESQ={pe:.4f}, SNR={ne:.2f}\n")

    
    if clean_fname == file_list[0]:  # only first file
        plot_waveforms(clean_np, noisy_np, enhanced_np, f"Waveforms - {clean_fname}")
        plot_spectrogram(clean_np, SR, "Clean Spectrogram")
        plot_spectrogram(noisy_np, SR, "Noisy Spectrogram")
        plot_spectrogram(enhanced_np, SR, "Enhanced Spectrogram")



print("\n===== AVERAGE RESULTS =====")
print(f"NOISY → CLEAN:    STOI={np.mean(noisy_stoi_list):.4f}, PESQ={np.mean(noisy_pesq_list):.4f}, SNR={np.mean(noisy_snr_list):.2f}")
print(f"ENHANCED → CLEAN: STOI={np.mean(enh_stoi_list):.4f}, PESQ={np.mean(enh_pesq_list):.4f}, SNR={np.mean(enh_snr_list):.2f}")



plot_average_metrics(noisy_stoi_list, noisy_pesq_list, noisy_snr_list,
                     enh_stoi_list, enh_pesq_list, enh_snr_list)

plot_per_file_improvements(noisy_stoi_list, enh_stoi_list, "STOI")
plot_per_file_improvements(noisy_pesq_list, enh_pesq_list, "PESQ")
plot_per_file_improvements(noisy_snr_list, enh_snr_list, "SNR")

plot_histograms(noisy_stoi_list, enh_stoi_list, "STOI")
plot_histograms(noisy_pesq_list, enh_pesq_list, "PESQ")
plot_histograms(noisy_snr_list, enh_snr_list, "SNR")
