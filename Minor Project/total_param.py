from unet import UNet
import torch

model = UNet()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {total_params:,}")
