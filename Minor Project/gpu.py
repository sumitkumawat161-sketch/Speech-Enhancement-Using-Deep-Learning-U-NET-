import torch

if torch.cuda.is_available():
    print(" CUDA available — GPU ready!")
    print("GPU:", torch.cuda.get_device_name())
else:
    print("Using CPU — no CUDA detected.")
