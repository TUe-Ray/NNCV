"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Add‑ons in this version:
- **FLOPs measurement** with `ptflops` (logged once after the final epoch).
- **Per‑class mean Dice scores** computed over the whole validation set (logged once after the final epoch).
- Both metrics are uploaded **in a single `wandb.Table`**, exactly once, after training finishes.
"""
import os
from argparse import ArgumentParser
from typing import Dict, List

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid

from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter,
    GaussianBlur,
    InterpolationMode,
    RandomCrop,
)

import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig

from ptflops import get_model_complexity_info  # NEW

# -----------------------------
# Utility helpers
# -----------------------------

# Cityscapes ID mappings
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    """Map raw Cityscapes IDs to train IDs in‑place."""
    return label_img.apply_(lambda x: id_to_trainid[x])


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image


# -----------------------------
# Argument parser
# -----------------------------

def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch SegFormer model with FLOPs & Dice logging")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (decoder) – encoder is scaled ×0.1")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="segformer-training", help="W&B run name")
    return parser


# -----------------------------
# Metrics helpers
# -----------------------------

def compute_flops(model: nn.Module, input_res: tuple) -> float:
    """Return FLOPs (not MACs) for a single forward pass."""
    macs, _ = get_model_complexity_info(
        model,
        input_res,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    flops = macs * 2  # 1 MAC = 2 FLOPs for conv/linear operations
    return flops


def dice_per_class(pred: torch.Tensor, target: torch.Tensor, n_cls: int) -> torch.Tensor:
    """Vectorised Dice per class for a single batch."""
    pred = pred.view(-1)
    target = target.view(-1)
    dice = torch.zeros(n_cls, device=pred.device, dtype=torch.float64)
    for c in range(n_cls):
        if c == 255:  # ignore label
            continue
        pred_c = pred == c
        tgt_c = target == c
        denom = pred_c.sum() + tgt_c.sum()
        if denom == 0:
            continue  # class absent in both pred & target
        inter = (pred_c & tgt_c).sum()
        dice[c] = 2 * inter / denom
    return dice


# -----------------------------
# Main training routine
# -----------------------------

def main(args):
    # W&B init
    wandb.init(
        project="5lsm0-cityscapes-segmentation-loss-combination",
        name=args.experiment_id,
        config=vars(args),
    )

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Dataset / Dataloaders ----------------
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

    train_transform = Compose([
        ToImage(), RandomHorizontalFlip(0.5), Resize((1024, 1024)),
        ColorJitter(0.3, 0.3, 0.3, 0.1), ToDtype(torch.float32, True),
        RandomRotation(30), GaussianBlur(3, (0.1, 2.0)),
        Normalize(processor.image_mean, processor.image_std),
    ])
    valid_transform = Compose([
        ToImage(), Resize((1024, 1024)), ToDtype(torch.float32, True),
        Normalize(processor.image_mean, processor.image_std),
    ])

    train_ds = wrap_dataset_for_transforms_v2(
        Cityscapes(args.data_dir, "train", "fine", "semantic", train_transform)
    )
    valid_ds = wrap_dataset_for_transforms_v2(
        Cityscapes(args.data_dir, "val", "fine", "semantic", valid_transform)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ---------------- Model ----------------
    config = SegformerConfig.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        num_channels=3,
        num_labels=19,
        upsample_ratio=1,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        config=config,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Loss & optimiser
    criterion = smp.losses.DiceLoss(mode="multiclass", ignore_index=255)
    dice_loss_fn = smp.losses.DiceLoss(mode="multiclass", ignore_index=255)

    enc_params, dec_params = [], []
    for n, p in model.named_parameters():
        (enc_params if "encoder" in n else dec_params).append(p)

    optimizer = AdamW([
        {"params": enc_params, "lr": args.lr * 0.1},
        {"params": dec_params, "lr": args.lr},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # Book‑keeping
    best_valid_loss = float("inf")
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- Training loop ----------------
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        # ----- train phase -----
        model.train()
        for i, (imgs, lbls) in enumerate(train_loader):
            lbls = convert_to_train_id(lbls)
            imgs, lbls = imgs.to(device), lbls.to(device)
            lbls = lbls.long().squeeze(1)

            optimizer.zero_grad()
            logits = model(imgs).logits
            logits = torch.nn.functional.interpolate(logits, size=lbls.shape[1:], mode="bilinear", align_corners=False)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[1]["lr"],
                "epoch": epoch + 1,
            }, step=epoch * len(train_loader) + i)

        # ----- validation phase -----
        model.eval()
        val_losses, val_dice_losses = [], []
        with torch.no_grad():
            for imgs, lbls in valid_loader:
                lbls = convert_to_train_id(lbls)
                imgs, lbls = imgs.to(device), lbls.to(device)
                lbls = lbls.long().squeeze(1)

                logits = model(imgs).logits
                logits = torch.nn.functional.interpolate(logits, size=lbls.shape[1:], mode="bilinear", align_corners=False)
                loss = criterion(logits, lbls)
                val_losses.append(loss.item())
                val_dice_losses.append(dice_loss_fn(logits, lbls).item())

        valid_loss = sum(val_losses) / len(val_losses)
        valid_dice_loss = sum(val_dice_losses) / len(val_dice_losses)
        scheduler.step()

        wandb.log({
            "valid_loss": valid_loss,
            "valid_dice_loss": valid_dice_loss,
        }, step=(epoch + 1) * len(train_loader) - 1)

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    print("Training complete 🏁")

    # -------------------------
    # FINAL METRICS (once!)
    # -------------------------
    model.eval()
    n_classes = 19
    dice_sum = torch.zeros(n_classes, dtype=torch.float64)
    n_batches = 0
    with torch.no_grad():
        for imgs, lbls in valid_loader:
            lbls = convert_to_train_id(lbls)
            imgs, lbls = imgs.to(device), lbls.to(device)
            lbls = lbls.long().squeeze(1)
            logits = model(imgs).logits
            logits = torch.nn.functional.interpolate(logits, size=lbls.shape[1:], mode="bilinear", align_corners=False)
            preds = logits.argmax(1)
            dice_sum += dice_per_class(preds, lbls, n_classes).cpu()
            n_batches += 1
    mean_dice = dice_sum / n_batches  # per class

    # FLOPs (single 1024×1024 forward)
    flops = compute_flops(model.to("cpu"), (3, 1024, 1024))
    flops_in_g = flops / 1e9

    # W&B table (metric, value)
    metrics_table = wandb.Table(columns=["Metric", "Value"])
    metrics_table.add_data("FLOPs (G)", f"{flops_in_g:.2f}")
    for cls in range(n_classes):
        metrics_table.add_data(f"Dice_class_{cls}", f"{mean_dice[cls]:.4f}")

    wandb.log({"FLOPs_and_MeanDice": metrics_table})
    wandb.finish()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
