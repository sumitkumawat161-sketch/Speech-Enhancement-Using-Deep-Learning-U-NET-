# ============================================================================
# continue_train.py — Continue Training Your Previous UNet Model
# ============================================================================

import torch
from torch import nn, optim
from tqdm import tqdm
import os

from dataloader_pt import get_loader
from preprocess.stft_utils import wav_to_mag_phase
from unet import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15        # continuing epochs
MAX_AUDIO_LEN = 64000
N_FFT = 512
HOP = 128
WIN = 512

PREVIOUS_MODEL = "models/unet_final.pt"     # <-- your saved model
NEW_NOISE_DIR = "data/mixed/noisy_5gb_4s"      # <-- your new noisy data
NEW_CLEAN_DIR = "data/mixed/clean_5gb_4s"      # <-- your new clean data

print("="*60)
print("🔥 Continuing Training")
print("Using:", DEVICE)
print("New data:", NEW_NOISE_DIR)
print("="*60)


train_loader = get_loader(
    noise_dir=NEW_NOISE_DIR,
    clean_dir=NEW_CLEAN_DIR,
    batch_size=BATCH_SIZE,
    max_len=MAX_AUDIO_LEN,
    num_workers=0
)

print(f"✓ Loaded {len(train_loader.dataset)} new samples")


model = UNet().to(DEVICE)
checkpoint = torch.load(PREVIOUS_MODEL, map_location=DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
print("✓ Loaded previous model weights")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
if "optimizer_state_dict" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("✓ Restored optimizer state")

criterion = nn.L1Loss()

start_epoch = checkpoint["epoch"]
print(f"Resuming from epoch {start_epoch}")


for epoch in range(start_epoch + 1, start_epoch + NUM_EPOCHS + 1):

    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for noisy, clean in pbar:

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        noisy_mag, noisy_phase = wav_to_mag_phase(noisy, N_FFT, HOP, WIN)
        clean_mag, _ = wav_to_mag_phase(clean, N_FFT, HOP, WIN)

        noisy_mag = noisy_mag.unsqueeze(1)
        clean_mag = clean_mag.unsqueeze(1)

        pred_mag = model(noisy_mag)

        
        min_f = min(pred_mag.shape[2], clean_mag.shape[2])
        min_t = min(pred_mag.shape[3], clean_mag.shape[3])

        pred_mag = pred_mag[:, :, :min_f, :min_t]
        clean_mag = clean_mag[:, :, :min_f, :min_t]

        loss = criterion(pred_mag, clean_mag)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

    
    save_path = f"models/unet_continue_epoch_{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }, save_path)
    print(f"✓ Saved: {save_path}")


torch.save({
    "epoch": start_epoch + NUM_EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": avg_loss,
}, "models/unet_final_continued.pt")

print("\n🎉 Continue Training Complete!")
print("Saved → models/unet_final_continued.pt")
