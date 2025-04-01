import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import timm

class SOTAUnet(nn.Module):
    def __init__(self, backbone="convnext_base.fb_in22k", in_channels=3, n_classes=19):
        super(SOTAUnet, self).__init__()
        
        # Load pre-trained ConvNeXt_base as encoder with proper weights
        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True)
        self.n_classes = n_classes

        # 使用更通用的解碼器結構，適應ConvNeXt的特徵通道數
        # 這裡我們會先檢測ConvNeXt的特徵尺寸，而不是硬編碼
        # 然後在forward函數中動態適應
        
        # 先設置基本的解碼器結構，通道數在forward中根據實際情況調整
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(0, 512, 256),  # 通道數將在forward中動態設置
            DecoderBlock(0, 256, 128),  # 通道數將在forward中動態設置
            DecoderBlock(0, 128, 64),   # 通道數將在forward中動態設置
            DecoderBlock(0, 64, 32)     # 通道數將在forward中動態設置
        ])
        
        # 最終分類層
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        
        # 為了調試，我們將運行一次前向傳播來獲取特徵尺寸
        self._initialize_feature_sizes()
        
    def _initialize_feature_sizes(self):
        # 通過一次前向傳播來獲取特徵尺寸
        with torch.no_grad():
            # 創建一個虛擬輸入以獲取特徵尺寸
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.encoder(dummy_input)
            
            # 打印特徵尺寸以進行調試
            print("Feature dimensions:")
            for i, feat in enumerate(features):
                print(f"Stage {i}: {feat.shape}")
                
            # 根據實際特徵尺寸重新創建解碼器
            feature_channels = [feat.shape[1] for feat in features]
            
            # 創建適合的解碼器
            self.decoder_blocks = nn.ModuleList([
                DecoderBlock(feature_channels[3], feature_channels[3]//2, feature_channels[3]//2),
                DecoderBlock(feature_channels[3]//2 + feature_channels[2], feature_channels[2]//2, feature_channels[2]//2),
                DecoderBlock(feature_channels[2]//2 + feature_channels[1], feature_channels[1]//2, feature_channels[1]//2),
                DecoderBlock(feature_channels[1]//2 + feature_channels[0], feature_channels[0]//2, feature_channels[0]//2)
            ])
            
            # 最終層應該使用最小的特徵通道數的一半
            self.final_conv = nn.Conv2d(feature_channels[0]//2, 19, kernel_size=1)
        
    def forward(self, x):
        original_size = x.shape[2:]  # ← 加在一開始
        features = self.encoder(x)  # 直接拿所有 stages

        
        # 解碼器前向傳播，使用跳躍連接
        # 從最深層開始
        x = features[3] # 最深層特徵
        x = self.decoder_blocks[0](x)
        x = F.interpolate(x, size=(features[2].shape[2], features[2].shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x, features[2]], dim=1)
        
        x = self.decoder_blocks[1](x)
        x = F.interpolate(x, size=(features[1].shape[2], features[1].shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x, features[1]], dim=1)
        
        x = self.decoder_blocks[2](x)
        x = F.interpolate(x, size=(features[0].shape[2], features[0].shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x, features[0]], dim=1)
        
        x = self.decoder_blocks[3](x)
        
        # 最後上採樣至原始輸入尺寸
        
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        # 最終分類
        x = self.final_conv(x)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 添加一些調試信息
        # print(f"DecoderBlock input shape: {x.shape}, expected channels: {self.in_channels}")
        return self.block(x)