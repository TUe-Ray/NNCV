import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

class ConvNeXtUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super(ConvNeXtUNet, self).__init__()
        
        # Load pre-trained ConvNeXt_base as encoder with proper weights
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        self.encoder = convnext_base(weights=weights)
        
        # Get the feature dimensions from ConvNeXt
        # ConvNeXt_base produces features with dimensions:
        # [96, 192, 384, 768]
        
        # Remove the classifier head
        self.encoder.classifier = nn.Identity()
        
        # Decoder path
        self.decoder4 = DecoderBlock(768, 384, 384)
        self.decoder3 = DecoderBlock(384 + 384, 192, 192)
        self.decoder2 = DecoderBlock(192 + 192, 96, 96)
        self.decoder1 = DecoderBlock(96 + 96, 48, 48)
        
        # Final classification layer
        self.final_conv = nn.Conv2d(48, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Store original input size for upsampling later
        original_size = (x.shape[2], x.shape[3])
        
        # Encoder forward - extract features at each stage
        features = []
        
        # ConvNeXt has stages in its features module
        # Stage 0 (stem)
        x = self.encoder.features[0](x)
        features.append(x)  # 96 channels
        
        # Stage 1
        x = self.encoder.features[1](x)
        features.append(x)  # 192 channels
        
        # Stage 2
        x = self.encoder.features[2](x)
        features.append(x)  # 384 channels
        
        # Stage 3
        x = self.encoder.features[3](x)
        features.append(x)  # 768 channels
        
        # Decoder forward with skip connections
        x = self.decoder4(x)
        x = F.interpolate(x, size=(features[2].shape[2], features[2].shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x, features[2]], dim=1)
        
        x = self.decoder3(x)
        x = F.interpolate(x, size=(features[1].shape[2], features[1].shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x, features[1]], dim=1)
        
        x = self.decoder2(x)
        x = F.interpolate(x, size=(features[0].shape[2], features[0].shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x, features[0]], dim=1)
        
        x = self.decoder1(x)
        
        # Final upsampling to original input size
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        # Final classification
        x = self.final_conv(x)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)