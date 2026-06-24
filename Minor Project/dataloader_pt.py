
import os
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

class SpeechDataset(Dataset):
    def __init__(self, noise_dir, clean_dir, max_len=64000):
        """
        Args:
            noise_dir: directory with noisy audio files
            clean_dir: directory with clean audio files
            max_len: maximum audio length (pad/trim to this)
        """
        self.noise_files = sorted([os.path.join(noise_dir, f) 
                                   for f in os.listdir(noise_dir) 
                                   if f.endswith(('.wav', '.flac'))])
        self.clean_files = sorted([os.path.join(clean_dir, f) 
                                   for f in os.listdir(clean_dir)
                                   if f.endswith(('.wav', '.flac'))])
        
        assert len(self.noise_files) == len(self.clean_files), \
            "Noise and clean directories must have same number of files"
        
        self.max_len = max_len

    def __len__(self):
        return len(self.noise_files)

    def __getitem__(self, idx):
        noisy, sr = sf.read(self.noise_files[idx])
        clean, sr = sf.read(self.clean_files[idx])

        # Convert to tensors
        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)

        # Pad or trim to fixed length
        noisy = self._fix_length(noisy, self.max_len)
        clean = self._fix_length(clean, self.max_len)

        return noisy, clean

    def _fix_length(self, audio, target_len):
        """Pad or trim audio to target length"""
        if len(audio) < target_len:
            # Pad with zeros
            padding = target_len - len(audio)
            audio = torch.cat([audio, torch.zeros(padding)])
        elif len(audio) > target_len:
            # Trim
            audio = audio[:target_len]
        return audio


def get_loader(noise_dir, clean_dir, batch_size=4, max_len=64000, num_workers=0):
    """
    Create DataLoader for speech enhancement.
    
    Args:
        noise_dir: path to noisy audio
        clean_dir: path to clean audio
        batch_size: batch size
        max_len: audio length (samples)
        num_workers: number of workers for data loading
    """
    dataset = SpeechDataset(noise_dir, clean_dir, max_len=max_len)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )

