import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SegFormerLike(nn.Module):
    def __init__(self, n_classes=19, backbone_name='mit_b2'):
        super(SegFormerLike, self).__init__()
        
        # 使用 timm 載入 MixVisionTransformer（MiT）系列當作 backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        
        # 假設會輸出 4 個特徵圖，從低解析度到高解析度
        self.embed_dims = [feat['num_chs'] for feat in self.backbone.feature_info]
        
        # 輕量解碼器：對每一層輸出做上採樣和線性處理後融合
        self.linear_c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for dim in self.embed_dims
        ])
        
        # 解碼後融合所有特徵（concatenated），再輸出 segmentation map
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        features = self.backbone(x)  # list of [B, C, H', W']
        
        upsampled = []
        for i in range(4):
            out = self.linear_c[i](features[i])  # [B, 256, h, w]
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            upsampled.append(out)
        
        fused = torch.cat(upsampled, dim=1)  # [B, 1024, H, W]
        out = self.fusion(fused)             # [B, n_classes, H, W]
        
        return out
