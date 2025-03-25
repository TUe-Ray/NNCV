import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedDiceCELoss(nn.Module):
    """
    將 CrossEntropyLoss 與 DiceLoss 做簡單加權相加的範例。
    適用於多類別分割。
    """
    def __init__(self, weight_dice=0.5, weight_ce=0.5, ignore_index=255):
        super(CombinedDiceCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, logits, targets):
        # logits: (N, C, H, W)
        # targets: (N, H, W)
        ce = self.ce_loss(logits, targets)
        dice = dice_loss(logits, targets, ignore_index=self.ignore_index)

        return self.weight_dice * dice + self.weight_ce * ce


# def dice_loss(logits, targets, ignore_index=255, eps=1e-6):
#     """
#     多類別 Dice Loss:
#     1. 先做 softmax 得到 (N, C, H, W) 的每類機率。
#     2. 將 targets 做 one-hot，忽略 ignore_index。
#     3. 分別計算每個類別的 Dice，最後取平均。
#     """
#     n, c, h, w = logits.shape
#     # 排除 ignore_index
#     valid_mask = (targets != ignore_index)  # (N, H, W)
#     # 預測機率
#     probs = F.softmax(logits, dim=1)  # (N, C, H, W) NxCxHxW 和目標 NxHxW，其中 N 是批次大小，C 是類別數

#     # ----- 關鍵修正 -----
#     # 1. 先複製一份 label，針對 ignore_index=255 的位置改成 0
#     targets_for_scatter = targets.clone()
#     targets_for_scatter[~valid_mask] = 0

#     # 2. 再做 scatter，不會發生 index out of range

#     # one-hot
#     with torch.no_grad():
#         targets_onehot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
#         # 忽略掉 ignore_index 的位置
#         targets_onehot = targets_onehot * valid_mask.unsqueeze(1)

#     # 對每個 class 分別計算 Dice
#     dims = (0, 2, 3)  # 從 batch, height, width 維度做 sum
#     intersection = torch.sum(probs * targets_onehot, dim=dims)
#     cardinality = torch.sum(probs, dim=dims) + torch.sum(targets_onehot, dim=dims) + eps
#     dice_per_class = 2. * intersection / cardinality

#     # 只計算有出現的類別，如果某個類別在整個 batch 都沒出現 (或被忽略)，
#     # 此處可能需要更進階的判斷。簡單做法是直接取所有類別平均。
#     return 1 - dice_per_class.mean()

def dice_loss(logits, targets, ignore_index=255, eps=1e-6):
    n, c, h, w = logits.shape
    # 建立有效 mask：只有非 ignore 的像素才有效
    valid_mask = (targets != ignore_index)  # (N, H, W)
    probs = F.softmax(logits, dim=1)       # (N, C, H, W)

    # 複製 targets 並把 ignore 區域設為 0
    targets_for_scatter = targets.clone()
    targets_for_scatter[~valid_mask] = 0

    # 檢查有效區域內的值是否都在 [0, c-1]
    assert targets_for_scatter.min() >= 0, "Negative label detected!"
    assert targets_for_scatter.max() < c, \
        f"Label value {targets_for_scatter.max().item()} out of range [0, {c-1}] detected! Check your convert_to_train_id mapping."

    # 使用 scatter 生成 one-hot
    with torch.no_grad():
        targets_onehot = torch.zeros_like(probs).scatter_(1, targets_for_scatter.unsqueeze(1), 1)

    # 再將 ignore 區域乘以 0
    targets_onehot = targets_onehot * valid_mask.unsqueeze(1)

    # 計算每個類別的 Dice
    dims = (0, 2, 3)  # sum over batch, height, width
    intersection = torch.sum(probs * targets_onehot, dim=dims)
    cardinality = torch.sum(probs, dim=dims) + torch.sum(targets_onehot, dim=dims) + eps
    dice_per_class = 2. * intersection / cardinality

    return 1 - dice_per_class.mean()

class FocalLoss(nn.Module):
    """
    多類別 Focal Loss 實作
    """
    def __init__(self, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)  # (N, H, W)
        pt = torch.exp(-ce)  # = 1 - p，p為對的機率
        focal_loss = ((1 - pt) ** self.gamma) * ce
        return focal_loss.mean()


class CombinedDiceFocalLoss(nn.Module):
    """
    將 Focal Loss 與 Dice Loss 加權相加
    """
    def __init__(self, weight_dice=0.5, weight_focal=0.5, gamma=2.0, ignore_index=255):
        super(CombinedDiceFocalLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.ignore_index = ignore_index
        self.focal_loss = FocalLoss(gamma=gamma, ignore_index=ignore_index)

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        dice = dice_loss(logits, targets, ignore_index=self.ignore_index)

        return self.weight_dice * dice + self.weight_focal * focal
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, ignore_index=255, eps=1e-6):
        """
        alpha > beta 時，懲罰假陽性 (FP) 更重
        alpha < beta 時，懲罰假陰性 (FN) 更重
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, targets):
        # logits: (N, C, H, W)
        # targets: (N, H, W)
        probs = F.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index)
        with torch.no_grad():
            targets_onehot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
            targets_onehot = targets_onehot * valid_mask.unsqueeze(1)

        dims = (0, 2, 3)
        tp = torch.sum(probs * targets_onehot, dims)
        fp = torch.sum(probs * (1 - targets_onehot), dims)
        fn = torch.sum((1 - probs) * targets_onehot, dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1 - tversky.mean()


class CombinedTverskyCELoss(nn.Module):
    """
    Tversky Loss 與 CrossEntropy Loss 的加權組合
    """
    def __init__(self, weight_tversky=0.5, weight_ce=0.5, alpha=0.7, beta=0.3, ignore_index=255):
        super(CombinedTverskyCELoss, self).__init__()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.weight_tversky = weight_tversky
        self.weight_ce = weight_ce

    def forward(self, logits, targets):
        loss_tv = self.tversky_loss(logits, targets)
        loss_ce = self.ce_loss(logits, targets)

        return self.weight_tversky * loss_tv + self.weight_ce * loss_ce
