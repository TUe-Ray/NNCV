import torch
import torch.nn as nn
from Model_Convnext_unet import ConvNeXtUNet

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 總參數量：{total:,}")
    print(f"🧠 可訓練參數量：{trainable:,}")
    return total, trainable

def count_encoder_decoder_params(model):
    encoder_params = 0
    decoder_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳過不需要訓練的參數
        if 'down' in name or 'enc' in name:
            encoder_params += param.numel()
        elif 'up' in name or 'dec' in name:
            decoder_params += param.numel()
    
    print(f"📥 Encoder 參數量：{encoder_params:,}")
    print(f"📤 Decoder 參數量：{decoder_params:,}")
    return encoder_params, decoder_params

# 建立模型
model = ConvNeXtUNet(in_channels=3, n_classes=19)

# 統計整體參數
print("📊 模型總參數統計：")
count_parameters(model)

# 統計 Encoder/Decoder
print("\n🔍 Encoder/Decoder 參數統計：")
count_encoder_decoder_params(model)
