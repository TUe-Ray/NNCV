import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class Model(nn.Module):
    def __init__(self, backbone_config="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
                 in_channels=3, n_classes=19):
        super(Model, self).__init__()

        # 從預訓練模型獲取設定，但不載入預訓練權重
        self.config = SegformerConfig.from_pretrained(
            backbone_config,
            num_channels=in_channels,
            num_labels=n_classes,
            upsample_ratio=1
        )

        # 使用 config 建立模型（不載入預訓練權重）
        self.segformer = SegformerForSemanticSegmentation(self.config)

        self.n_classes = n_classes

    def forward(self, x):
        output = self.segformer(x)
        return output.logits  # logits 是語義分割的輸出
