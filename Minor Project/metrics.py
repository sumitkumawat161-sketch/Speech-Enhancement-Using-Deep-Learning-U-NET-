import numpy as np
from pystoi import stoi
from pesq import pesq

def calculate_metrics(clean, enhanced, sr=16000):

    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    stoi_score = stoi(clean, enhanced, sr, extended=False)
    pesq_score = pesq(sr, clean, enhanced, 'wb')

    # SNR
    noise = clean - enhanced
    snr_score = 10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2))

    return stoi_score, pesq_score, snr_score
