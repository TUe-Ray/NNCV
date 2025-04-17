import os
from argparse import ArgumentParser

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


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation-loss-combination",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

    train_transform = Compose([
        ToImage(),
        RandomHorizontalFlip(p=0.5),
        Resize((1024, 1024)),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ToDtype(torch.float32, scale=True),
        RandomRotation(degrees=30),
        GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    valid_transform = Compose([
        ToImage(),
        Resize((1024, 1024)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=train_transform
    )
    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transforms=valid_transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    config = SegformerConfig.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        num_channels=3,
        num_labels=19,
        upsample_ratio=1
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        config=config,
        ignore_mismatched_sizes=True
    ).to(device)

    # Number of classes for per-category dice
    num_classes = model.config.num_labels
    # Prepare class names by train_id order
    class_names = [None] * num_classes
    for cls in Cityscapes.classes:
        if cls.train_id < num_classes:
            class_names[cls.train_id] = cls.name

    criterion = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
    dice_loss_fn = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)

    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},
        {'params': decoder_params, 'lr': args.lr}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    best_valid_loss = float('inf')
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=labels.shape[1:], mode='bilinear', align_corners=False
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[1]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            dice_losses = []
            # Prepare accumulators for last-epoch per-class dice
            if epoch == args.epochs - 1:
                inters = torch.zeros(num_classes, device=device)
                preds_sum = torch.zeros(num_classes, device=device)
                labels_sum = torch.zeros(num_classes, device=device)

            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                outputs = model(images)
                logits = outputs.logits
                logits = torch.nn.functional.interpolate(
                    logits, size=labels.shape[1:], mode='bilinear', align_corners=False
                )
                loss = criterion(logits, labels)
                losses.append(loss.item())

                dice_loss_val = dice_loss_fn(logits, labels)
                dice_losses.append(dice_loss_val.item())

                # accumulate per-class counts on final epoch
                if epoch == args.epochs - 1:
                    preds = logits.softmax(1).argmax(1)
                    for c in range(num_classes):
                        p_mask = preds == c
                        l_mask = labels == c
                        inters[c] += (p_mask & l_mask).sum()
                        preds_sum[c] += p_mask.sum()
                        labels_sum[c] += l_mask.sum()

                if i == 0:
                    predictions = logits.softmax(1).argmax(1).unsqueeze(1)
                    gt_labels = labels.unsqueeze(1)
                    preds_color = convert_train_id_to_color(predictions)
                    labels_color = convert_train_id_to_color(gt_labels)
                    wandb.log({
                        "predictions": [wandb.Image(make_grid(preds_color.cpu(), nrow=8).permute(1,2,0).numpy())],
                        "labels": [wandb.Image(make_grid(labels_color.cpu(), nrow=8).permute(1,2,0).numpy())],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)

            valid_loss = sum(losses) / len(losses)
            valid_dice_loss = sum(dice_losses) / len(dice_losses)

            scheduler.step()
            print(f"[Epoch {epoch+1}] LR encoder: {optimizer.param_groups[0]['lr']:.6f}, decoder: {optimizer.param_groups[1]['lr']:.6f}")

            wandb.log({
                "valid_loss": valid_loss,
                "valid_dice_loss": valid_dice_loss,
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)

            # After last epoch validation, compute and print per-class mean dice
            if epoch == args.epochs - 1:
                dice_per_class = (2 * inters) / (preds_sum + labels_sum + 1e-8)
                # Console output as markdown table
                print("\nMean Dice per Category (Final Epoch):")
                print("| Class | Mean Dice |")
                print("|---|---|")
                for idx, name in enumerate(class_names):
                    print(f"| {name} | {dice_per_class[idx].item():.4f} |")
                # Log to wandb as a table
                table_data = [[class_names[i], float(dice_per_class[i].item())] for i in range(num_classes)]
                table = wandb.Table(data=table_data, columns=["class","mean_dice"])
                wandb.log({"mean_dice_per_category": table})

    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
