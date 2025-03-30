import torch
import torch.nn as nn
from unet import UNet

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 總參數量：{total:,}")
    print(f"🧠 可訓練參數量：{trainable:,}")
    return total, trainable

# 範例：建立 ResUNet 並計算參數量
model = UNet(in_channels=3, n_classes=19)
count_parameters(model)
