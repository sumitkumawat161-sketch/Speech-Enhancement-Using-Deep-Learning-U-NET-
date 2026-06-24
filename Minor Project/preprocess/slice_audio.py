import os
import soundfile as sf
import numpy as np

def slice_audio(input_clean, input_noisy, output_clean, output_noisy, segment_seconds=4, sr=16000):
    os.makedirs(output_clean, exist_ok=True)
    os.makedirs(output_noisy, exist_ok=True)

    clean_files = sorted([f for f in os.listdir(input_clean) if f.endswith('.wav')])
    noisy_files = sorted([f for f in os.listdir(input_noisy) if f.endswith('.wav')])

    assert len(clean_files) == len(noisy_files), "Mismatch: clean/noisy count"

    segment_len = segment_seconds * sr

    for clean_f, noisy_f in zip(clean_files, noisy_files):
        clean_path = os.path.join(input_clean, clean_f)
        noisy_path = os.path.join(input_noisy, noisy_f)

        clean_audio, _ = sf.read(clean_path)
        noisy_audio, _ = sf.read(noisy_path)

        for i in range(0, len(clean_audio), segment_len):
            c = clean_audio[i:i + segment_len]
            n = noisy_audio[i:i + segment_len]

            # pad if short
            if len(c) < segment_len:
                c = np.pad(c, (0, segment_len - len(c)))
                n = np.pad(n, (0, segment_len - len(n)))

            clean_out = os.path.join(output_clean, f"{clean_f[:-4]}_{i}.wav")
            noisy_out = os.path.join(output_noisy, f"{noisy_f[:-4]}_{i}.wav")

            sf.write(clean_out, c, sr)
            sf.write(noisy_out, n, sr)

if __name__ == "__main__":
    slice_audio(
        input_clean="data/mixed/clean_5gb",
        input_noisy="data/mixed/noisy_5gb",
        output_clean="data/mixed/clean_5gb_4s",
        output_noisy="data/mixed/noisy_5gb_4s"
    )
