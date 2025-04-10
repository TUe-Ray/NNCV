
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
)

import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SegformerImageProcessor

# Import your custom model
from test_model import Model

id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch segmentation model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-id", type=str, default="custom-model-training", help="Experiment ID for wandb")

    return parser


def main(args):
    wandb.init(project="5lsm0-cityscapes-segmentation-loss-combination", name=args.experiment_id, config=vars(args))

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

    train_transform = Compose([
        ToImage(),
        RandomHorizontalFlip(p=0.5),
        Resize((512, 512)),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ToDtype(torch.float32, scale=True),
        RandomRotation(degrees=30),
        GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    valid_transform = Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    train_dataset = wrap_dataset_for_transforms_v2(Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=train_transform))
    valid_dataset = wrap_dataset_for_transforms_v2(Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=valid_transform))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load your custom model with weights
    model = Model(in_channels=3, n_classes=19)
    model.load_state_dict(torch.load("test_model.pth", map_location=device))
    model.to(device)

    criterion = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_dataloader:
            labels = convert_to_train_id(labels).long().squeeze(1).to(device)
            images = images.to(device)

            optimizer.zero_grad()
            logits = model(images)
            logits = torch.nn.functional.interpolate(logits, size=labels.shape[1:], mode='bilinear', align_corners=False)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item(), "epoch": epoch + 1})

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, labels in valid_dataloader:
                labels = convert_to_train_id(labels).long().squeeze(1).to(device)
                images = images.to(device)

                logits = model(images)
                logits = torch.nn.functional.interpolate(logits, size=labels.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(logits, labels)
                valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)
        scheduler.step()

        wandb.log({"valid_loss": valid_loss}, step=(epoch + 1))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
