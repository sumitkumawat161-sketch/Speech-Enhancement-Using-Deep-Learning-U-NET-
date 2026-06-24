# Speech Enhancement Using Deep Learning U-Net

A PyTorch-based Speech Enhancement System that removes background noise from speech signals using a U-Net architecture. The model processes noisy audio through STFT-based spectrogram representations, enhances speech quality using deep learning, and reconstructs clean audio with ISTFT. The system is trained and evaluated using industry-standard metrics such as STOI, PESQ, and SNR.

##  Features

* Speech Denoising using U-Net Architecture
* STFT & ISTFT Based Audio Processing
* Encoder-Decoder Network with Skip Connections
* Custom PyTorch Dataset & DataLoader
* GPU Acceleration Support (CUDA)
* Evaluation using STOI, PESQ, and SNR Metrics

##  Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib

##  Datasets

* VoiceBank-DEMAND
* LibriSpeech
* MUSAN

## Workflow

```text
Noisy Audio
     ↓
STFT
     ↓
Magnitude Spectrogram
     ↓
U-Net Model
     ↓
Enhanced Spectrogram
     ↓
ISTFT
     ↓
Enhanced Audio
```

##  Model Architecture

The model uses a U-Net architecture consisting of:

* Encoder for feature extraction
* Bottleneck for learning complex noise patterns
* Decoder for speech reconstruction
* Skip Connections for preserving speech details

##  Evaluation Metrics

* STOI (Speech Intelligibility)
* PESQ (Speech Quality)
* SNR (Signal-to-Noise Ratio)



## Applications

* Voice Assistants
* Video Conferencing
* Call Centers
* Hearing Aids
* Speech Recognition Systems
* Mobile Communication

## Future Improvements

* Transformer-based Speech Enhancement
* GAN-based Denoising
* Real-Time Processing
* Phase-Aware Enhancement
* Web Deployment

## 👨‍💻 Author

**Sumit Kumar Kumawat**

B.Tech, Electronics & Communication Engineering

