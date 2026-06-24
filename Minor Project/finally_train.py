import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
...

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

from unet import UNet
from preprocess.stft_utils import wav_to_mag_phase, mag_phase_to_wav
from metrics import calculate_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "models/unet_final_continued.pt"
clean_file = "test_audio/clean.wav"
noisy_file = "test_audio/noisy.wav"

# Load model
model = UNet().to(DEVICE)
checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load audio
clean, sr = torchaudio.load(clean_file)
noisy, sr = torchaudio.load(noisy_file)

clean = clean.to(DEVICE)
noisy = noisy.to(DEVICE)

# STFT
mag, phase = wav_to_mag_phase(noisy)
mag = mag.unsqueeze(1)

# Enhance
with torch.no_grad():
    enhanced_mag = model(mag)

enhanced_mag = enhanced_mag.squeeze(1)
# Convert to waveform
enhanced = mag_phase_to_wav(enhanced_mag, phase)
# Crop both to same size
# Correct cropping based on waveform length
min_len = min(clean.shape[-1], enhanced.shape[-1])

clean = clean[:, :min_len]
enhanced = enhanced[:, :min_len]


clean_np = clean.squeeze().cpu().numpy()
enhanced_np = enhanced.squeeze().cpu().numpy()


# ======== METRICS ========
stoi_score, pesq_score, snr_score = calculate_metrics(clean_np, enhanced_np)

print("===================================")
print(f"STOI Score: {stoi_score:.4f}")
print(f"PESQ Score: {pesq_score:.4f}")
print(f"SNR Score : {snr_score:.2f} dB")
print("===================================")

# ======== GRAPH ========

plt.figure(figsize=(12,6))
plt.plot(clean[0].cpu(), label="Clean", alpha=0.7)
plt.plot(noisy[0].cpu(), label="Noisy", alpha=0.7)
plt.plot(enhanced[0].cpu(), label="Enhanced", alpha=0.7)
plt.legend()
plt.title("Waveform Comparison")
plt.tight_layout()
plt.show()
