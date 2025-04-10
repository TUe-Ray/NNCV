import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class Model(SegformerForSemanticSegmentation):
    def __init__(self, backbone_config="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
                 in_channels=3, n_classes=19):
        config = SegformerConfig.from_pretrained(
            backbone_config,
            num_channels=in_channels,
            num_labels=n_classes,
            upsample_ratio=1
        )
        super(Model, self).__init__(config)

    def forward(self, x):
        output = super(Model, self).forward(x)
        return output.logits  # logits 是語義分割的輸出
