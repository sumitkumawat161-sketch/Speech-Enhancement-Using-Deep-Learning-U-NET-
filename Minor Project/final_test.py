import torch
import torchaudio

from unet import UNet
from preprocess.stft_utils import wav_to_mag_phase, mag_phase_to_wav



model_path = "models/unet_final_continued.pt"        
input_wav_path = "test_audio/noisy.wav"    
output_wav_path = "test_audio/clean_output.wav"
sr = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = UNet().to(device)


checkpoint = torch.load(model_path, map_location=device)    
model.load_state_dict(checkpoint["model_state_dict"])
model.eval() # Inference Mode

waveform, file_sr = torchaudio.load(input_wav_path)
waveform = torchaudio.functional.resample(waveform, file_sr, sr)
waveform = waveform.to(device)


mag, phase = wav_to_mag_phase(waveform, n_fft=512, hop=128, win=512)


mag_input = mag.unsqueeze(1)


with torch.no_grad():
    enhanced_mag = model(mag_input)


enhanced_mag = enhanced_mag.squeeze(1)


enhanced_waveform = mag_phase_to_wav(
    enhanced_mag,
    phase,
    n_fft=512,
    hop=128,
    win=512
)


if enhanced_waveform.dim() == 1:
    enhanced_waveform = enhanced_waveform.unsqueeze(0)


torchaudio.save(output_wav_path, enhanced_waveform.cpu(), sr)

print(" Cleaned audio saved at:", output_wav_path)
