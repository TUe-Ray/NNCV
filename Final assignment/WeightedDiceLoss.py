from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from ._functional import soft_dice_score, to_tensor
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["WeightedDiceLoss"]


class WeightedDiceLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        class_weights: Optional[torch.Tensor] = None,
    ):
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(WeightedDiceLoss, self).__init__()
        self.mode = mode

        if classes is not None:
            assert mode != BINARY_MODE, (
                "Masking classes is not supported with mode=binary"
            )
            classes = to_tensor(classes, dtype=torch.long)

        if class_weights is not None:
            class_weights = to_tensor(class_weights, dtype=torch.float)

        self.classes = classes
        self.class_weights = class_weights
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes)
                y_true = y_true.permute(0, 2, 1)

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.class_weights is not None:
            loss = loss * self.class_weights.to(loss.device)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)
