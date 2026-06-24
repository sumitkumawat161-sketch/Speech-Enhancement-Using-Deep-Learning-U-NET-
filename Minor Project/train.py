import torch
from torch import nn, optim
from tqdm import tqdm
import os

from preprocess.stft_utils import wav_to_mag_phase, mag_phase_to_wav
from dataloader_pt import get_loader
from unet import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
MAX_AUDIO_LEN = 64000  
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")
print("=" * 60)


try:
    train_loader = get_loader(
        noise_dir="data/noise_4s",
        clean_dir="data/clean_4s",
        batch_size=BATCH_SIZE,
        max_len=MAX_AUDIO_LEN,
        num_workers=0  
    )
    print(f" Dataset loaded: {len(train_loader.dataset)} samples")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    exit(1)


model = UNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.L1Loss()


total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model initialized: {total_params:,} parameters")


os.makedirs("models", exist_ok=True)


print("\n" + "=" * 60)
print("Starting Training")
print("=" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    
    for batch_idx, (noisy, clean) in enumerate(pbar):
        
        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)
        
        
        noisy_mag, noisy_phase = wav_to_mag_phase(
            noisy, n_fft=N_FFT, hop=HOP_LENGTH, win=WIN_LENGTH
        )
        clean_mag, _ = wav_to_mag_phase(
            clean, n_fft=N_FFT, hop=HOP_LENGTH, win=WIN_LENGTH
        )
        
        
        noisy_mag = noisy_mag.unsqueeze(1)
        clean_mag = clean_mag.unsqueeze(1)
        
        
        pred_mag = model(noisy_mag)
        
        
        if pred_mag.shape != clean_mag.shape:
            min_h = min(pred_mag.shape[2], clean_mag.shape[2])
            min_w = min(pred_mag.shape[3], clean_mag.shape[3])
            pred_mag = pred_mag[:, :, :min_h, :min_w]
            clean_mag = clean_mag[:, :, :min_h, :min_w]
        
        
        loss = criterion(pred_mag, clean_mag)
        
        
        optimizer.zero_grad()
        loss.backward()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch:2d} | Average Loss: {avg_loss:.4f}")
    
    
    if epoch % 5 == 0:
        checkpoint_path = f"models/unet_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")


final_model_path = "models/unet_final.pt"
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, final_model_path)

print("\n" + "=" * 60)
print(f"✓ Training complete! Model saved to: {final_model_path}")
print("=" * 60)