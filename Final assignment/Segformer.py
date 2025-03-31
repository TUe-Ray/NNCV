import os
from argparse import ArgumentParser

import wandb
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid

from torchvision.transforms.v2 import (
    Compose, Normalize, Resize, ToImage, ToDtype,
    RandomHorizontalFlip, RandomRotation, ColorJitter, GaussianBlur
)

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image

def get_args_parser():
    parser = ArgumentParser("Training script for SegFormer")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="segformer-training")
    return parser

def main(args):
    wandb.init(project="cityscapes-segformer", name=args.experiment_id, config=vars(args))
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")

    train_transform = Compose([
        ToImage(), RandomHorizontalFlip(), Resize((512, 512)), ColorJitter(0.3,0.3,0.3,0.1),
        RandomRotation(15), GaussianBlur(3), ToDtype(torch.float32, scale=True),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    valid_transform = Compose([
        ToImage(), Resize((512, 512)), ToDtype(torch.float32, scale=True),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    train_dataset = wrap_dataset_for_transforms_v2(Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=train_transform))
    valid_dataset = wrap_dataset_for_transforms_v2(Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=valid_transform))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        num_labels=19, ignore_mismatched_sizes=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_dataloader:
            labels = convert_to_train_id(labels).squeeze(1).long().to(device)
            images = images.to(device)

            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item(), "epoch": epoch})

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels).squeeze(1).long().to(device)
                images = images.to(device)

                outputs = model(pixel_values=images, labels=labels)
                total_loss += outputs.loss.item()

                if idx == 0:
                    preds = outputs.logits.argmax(dim=1).cpu()
                    pred_colors = convert_train_id_to_color(preds)
                    pred_grid = make_grid(pred_colors, nrow=4).permute(1,2,0)
                    wandb.log({"predictions": wandb.Image(pred_grid.numpy())}, step=epoch)

            avg_loss = total_loss / len(valid_dataloader)
            wandb.log({"valid_loss": avg_loss}, step=epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_epoch_{epoch}.pth"))

    wandb.finish()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
