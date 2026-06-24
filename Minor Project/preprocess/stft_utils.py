import torch

def wav_to_mag_phase(waveform, n_fft=512, hop=128, win=512):
    """
    Convert waveform to magnitude and phase spectrograms.
    waveform: [B, T] tensor
    returns: mag [B, F, T], phase [B, F, T]
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(win, device=waveform.device)

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        return_complex=True
    )  # [B, F, T]

    mag = torch.abs(stft)
    phase = torch.angle(stft)

    return mag, phase


def mag_phase_to_wav(mag, phase, n_fft=512, hop=128, win=512, target_len=None):
    """
    Reconstruct waveform from magnitude and phase.
    mag: [B, F, T]
    phase: [B, F, T]
    returns: wav [B, T]
    """
    stft_complex = torch.polar(mag, phase)
    window = torch.hann_window(win, device=mag.device)

    wav = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        length=target_len
    )

    return wav
