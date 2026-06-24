from preprocess.stft_utils import wav_to_mag_phase
import os

test_file = "data/noise_4s/p226_001_0.wav"

# check file exists
if not os.path.exists(test_file):
    print("File not found:", test_file)
    exit()

mag, phase = wav_to_mag_phase(test_file)

print("STFT SUCCESSFUL!")
print("Magnitude shape:", mag.shape)
print("Phase shape:", phase.shape)
print("Example magnitude values:", mag.flatten()[:10])
