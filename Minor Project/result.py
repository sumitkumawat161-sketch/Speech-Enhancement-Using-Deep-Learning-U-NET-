import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import random
import shutil
import torch
import torchaudio
import numpy as np

from unet import UNet
from preprocess.stft_utils import wav_to_mag_phase, mag_phase_to_wav
from metrics import calculate_metrics  # must accept numpy arrays

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
NUM_TEST_FILES = 30

clean_dir = "data/mixed/clean/"
noisy_dir = "data/mixed/noisy/"

test_clean_dir = "test_set/clean/"
test_noisy_dir = "test_set/noisy/"

model_path = "models/unet_final_continued.pt"

# -----------------------------
# CREATE TEST SET FOLDERS
# -----------------------------
os.makedirs(test_clean_dir, exist_ok=True)
os.makedirs(test_noisy_dir, exist_ok=True)

# -----------------------------
# LIST FILES AND MATCH BY NUMBER
# -----------------------------
clean_files = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]
noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith(".wav")]

# Build dict of noisy files by initial number
noisy_dict = {f.split("_")[0]: f for f in noisy_files}

# Only keep clean files that have a matching noisy file
matched_clean_files = [f for f in clean_files if f.split("_")[0] in noisy_dict]

# Randomly select subset
selected_clean_files = random.sample(matched_clean_files, min(NUM_TEST_FILES, len(matched_clean_files)))

# -----------------------------
# COPY SELECTED FILES TO TEST SET
# -----------------------------
for clean_fname in selected_clean_files:
    num = clean_fname.split("_")[0]
    noisy_fname = noisy_dict[num]

    shutil.copy(os.path.join(clean_dir, clean_fname), os.path.join(test_clean_dir, clean_fname))
    shutil.copy(os.path.join(noisy_dir, noisy_fname), os.path.join(test_noisy_dir, noisy_fname))

print(f"\nCopied {len(selected_clean_files)} matched clean/noisy file pairs to test_set/.\n")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = UNet().to(DEVICE)
checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -----------------------------
# METRICS STORAGE
# -----------------------------
noisy_stoi_list = []
noisy_pesq_list = []
noisy_snr_list = []

enh_stoi_list = []
enh_pesq_list = []
enh_snr_list = []

# -----------------------------
# ENHANCEMENT FUNCTION
# -----------------------------
def enhance(noisy_tensor):
    mag, phase = wav_to_mag_phase(noisy_tensor)
    mag = mag.unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        enhanced_mag = model(mag)
    enhanced_mag = enhanced_mag.squeeze(1)
    enhanced = mag_phase_to_wav(enhanced_mag, phase)
    return enhanced

# -----------------------------
# EVALUATE TEST SET
# -----------------------------
file_list = sorted(os.listdir(test_clean_dir))
print(f"Evaluating {len(file_list)} files...\n")

for clean_fname in file_list:
    num = clean_fname.split("_")[0]
    noisy_fname = noisy_dict[num]

    clean_path = os.path.join(test_clean_dir, clean_fname)
    noisy_path = os.path.join(test_noisy_dir, noisy_fname)

    clean, _ = torchaudio.load(clean_path)
    noisy, _ = torchaudio.load(noisy_path)

    clean = clean.to(DEVICE)
    noisy = noisy.to(DEVICE)

    # Enhance
    enhanced = enhance(noisy)

    # Crop lengths to smallest
    min_len = min(clean.shape[-1], noisy.shape[-1], enhanced.shape[-1])
    clean = clean[:, :min_len]
    noisy = noisy[:, :min_len]
    enhanced = enhanced[:, :min_len]

    # Convert to numpy
    clean_np = clean.squeeze().cpu().numpy()
    noisy_np = noisy.squeeze().cpu().numpy()
    enhanced_np = enhanced.squeeze().cpu().numpy()

    # ----- NOISY vs CLEAN -----
    stoi_n, pesq_n, snr_n = calculate_metrics(clean_np, noisy_np)
    noisy_stoi_list.append(stoi_n)
    noisy_pesq_list.append(pesq_n)
    noisy_snr_list.append(snr_n)

    # ----- ENHANCED vs CLEAN -----
    stoi_e, pesq_e, snr_e = calculate_metrics(clean_np, enhanced_np)
    enh_stoi_list.append(stoi_e)
    enh_pesq_list.append(pesq_e)
    enh_snr_list.append(snr_e)

    print(f"{clean_fname}:")
    print(f"  Noisy     -> STOI={stoi_n:.4f}, PESQ={pesq_n:.4f}, SNR={snr_n:.2f} dB")
    print(f"  Enhanced  -> STOI={stoi_e:.4f}, PESQ={pesq_e:.4f}, SNR={snr_e:.2f} dB\n")

# -----------------------------
# FINAL AVERAGES
# -----------------------------
print("\n===== FINAL AVERAGE RESULTS =====\n")
print("NOISY vs CLEAN:")
print(f"  Avg STOI: {np.mean(noisy_stoi_list):.4f}")
print(f"  Avg PESQ: {np.mean(noisy_pesq_list):.4f}")
print(f"  Avg SNR : {np.mean(noisy_snr_list):.2f} dB\n")

print("ENHANCED vs CLEAN:")
print(f"  Avg STOI: {np.mean(enh_stoi_list):.4f}")
print(f"  Avg PESQ: {np.mean(enh_pesq_list):.4f}")
print(f"  Avg SNR : {np.mean(enh_snr_list):.2f} dB\n")

print("IMPROVEMENTS:")
print(f"  STOI Improvement: {np.mean(enh_stoi_list) - np.mean(noisy_stoi_list):.4f}")
print(f"  PESQ Improvement: {np.mean(enh_pesq_list) - np.mean(noisy_pesq_list):.4f}")
print(f"  SNR Improvement : {np.mean(enh_snr_list) - np.mean(noisy_snr_list):.2f} dB")
