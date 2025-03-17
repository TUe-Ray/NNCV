import torch
import torch.nn as nn
import torchvision.models as models


class ResUNet(nn.Module):
    """ 
    A simple U-Net architecture for image segmentation.
    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, in_channels=3, n_classes=19):
        
        super(ResUNet, self).__init__()
        # 載入預訓練的 ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

         # Encoder 部分：提取 ResNet34 前幾個模組作為特徵抽取器
        self.encoder0 = nn.Sequential(
            resnet.conv1,  # 輸出通道：64，尺寸：原始尺寸/2 =128
            resnet.bn1,
            resnet.relu
        )
        self.encoder1 = nn.Sequential(
            resnet.maxpool,  # 下採樣，使尺寸變為原始尺寸/4
            resnet.layer1   # 輸出通道：64
        )
        self.encoder2 = resnet.layer2   # 輸出通道：128，尺寸：/8
        self.encoder3 = resnet.layer3   # 輸出通道：256，尺寸：/16
        self.encoder4 = resnet.layer4   # 輸出通道：512，尺寸：/32 =8

         # up4: 融合 encoder4 (512 channels) 與 encoder3 (256 channels) output = 16
        self.up4 = Up(in_channels=512 + 256, out_channels=256)
        # up3: 融合上一層輸出 (256 channels) 與 encoder2 (128 channels) output = 32
        self.up3 = Up(in_channels=256 + 128, out_channels=128)
        # up2: 融合上一層輸出 (128 channels) 與 encoder1 (64 channels) output = 64
        self.up2 = Up(in_channels=128 + 64, out_channels=64)
        # up1: 融合上一層輸出 (64 channels) 與 encoder0 (64 channels)output = 128
        self.up1 = Up(in_channels=64 + 64, out_channels=64)

        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 最後輸出層：將通道數轉換為分類數
        self.outc = OutConv(in_channels=64, out_channels=n_classes)

    def forward(self, x):
         # Encoder 部分
        x0 = self.encoder0(x)   # 輸出尺寸：/2
        x1 = self.encoder1(x0)  # 輸出尺寸：/4
        x2 = self.encoder2(x1)  # 輸出尺寸：/8
        x3 = self.encoder3(x2)  # 輸出尺寸：/16
        x4 = self.encoder4(x3)  # 輸出尺寸：/32

        # Decoder 部分，透過上採樣並與 encoder 特徵做 skip connection
        d4 = self.up4(x4, x3)   # 融合 encoder4 與 encoder3
        d3 = self.up3(d4, x2)   # 融合上一層與 encoder2
        d2 = self.up2(d3, x1)   # 融合上一層與 encoder1
        d1 = self.up1(d2, x0)   # 融合上一層與 encoder0
        d0 = self.upsampling(d1)     # 上採樣回原始尺寸
        logits = self.outc(d0) # 輸出分類結果

        return logits
        

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)





class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # 使用 bilinear upsampling 進行尺寸擴大
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 注意：in_channels 為 concat 後的通道數，因此 DoubleConv 的 mid_channels 設為 in_channels//2
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 假設 x1 與 x2 的尺寸相同；若有尺寸差異，可能需要裁剪或 padding
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)