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
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig

torch.backends.cudnn.benchmark = True  # Enable if inputs have fixed size

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
    parser = ArgumentParser("Training script with mixed precision and per-category Dice metrics")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-mixprec", help="WandB experiment ID")
    return parser


def main(args):
    # Initialize wandb
    wandb.init(project="5lsm0-cityscapes-segmentation-loss-combination", name=args.experiment_id, config=vars(args))

    # Output dir
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Processor and transforms
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    train_transform = Compose([
        ToImage(), RandomHorizontalFlip(0.5), Resize((1024, 1024)), ColorJitter(0.3,0.3,0.3,0.1),
        ToDtype(torch.float32, scale=True), RandomRotation(30), GaussianBlur(3,(0.1,2.0)),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    valid_transform = Compose([
        ToImage(), Resize((1024,1024)), ToDtype(torch.float32, scale=True), Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    # Datasets and loaders
    train_ds = wrap_dataset_for_transforms_v2(Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=train_transform))
    valid_ds = wrap_dataset_for_transforms_v2(Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=valid_transform))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model setup
    config = SegformerConfig.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024", num_channels=3, num_labels=19, upsample_ratio=1)
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024", config=config, ignore_mismatched_sizes=True
    ).to(device)

    # Class names
    num_classes = model.config.num_labels
    class_names = [None] * num_classes
    for cls in Cityscapes.classes:
        if cls.train_id < num_classes:
            class_names[cls.train_id] = cls.name

    # Losses
    criterion = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
    dice_fn = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)

    # Optimizer & scheduler
    enc_params, dec_params = [], []
    for n,p in model.named_parameters(): (enc_params if 'encoder' in n else dec_params).append(p)
    optimizer = AdamW([{'params':enc_params,'lr':args.lr*0.1},{'params':dec_params,'lr':args.lr}])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    best_valid = float('inf')
    best_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        for i,(imgs, lbls) in enumerate(train_loader):
            lbls = convert_to_train_id(lbls).long().squeeze(1).to(device)
            imgs = imgs.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(imgs).logits
                out = torch.nn.functional.interpolate(out, size=lbls.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            wandb.log({"train_loss": loss.item(), "lr": optimizer.param_groups[1]['lr'], "epoch": epoch+1}, step=epoch*len(train_loader)+i)

        # Validation
        model.eval()
        val_losses, val_dices = [], []
        if epoch == args.epochs-1:
            inters = torch.zeros(num_classes, device=device); preds_sum = torch.zeros_like(inters); labels_sum = torch.zeros_like(inters)

        with torch.no_grad():
            for j,(imgs, lbls) in enumerate(valid_loader):
                lbls = convert_to_train_id(lbls).long().squeeze(1).to(device)
                imgs = imgs.to(device)
                with torch.cuda.amp.autocast():
                    out = model(imgs).logits
                    out = torch.nn.functional.interpolate(out, size=lbls.shape[1:], mode='bilinear', align_corners=False)
                    lossv = criterion(out, lbls)
                val_losses.append(lossv.item())
                val_dices.append(dice_fn(out, lbls).item())

                if epoch == args.epochs-1:
                    pred = out.softmax(1).argmax(1)
                    for c in range(num_classes):
                        pm = pred==c; lm = lbls==c
                        inters[c] += (pm & lm).sum(); preds_sum[c] += pm.sum(); labels_sum[c] += lm.sum()

                if j==0:
                    p_color = convert_train_id_to_color(pred.unsqueeze(1)); l_color = convert_train_id_to_color(lbls.unsqueeze(1))
                    wandb.log({"preds":[wandb.Image(make_grid(p_color.cpu(),nrow=8).permute(1,2,0).numpy())],"gts":[wandb.Image(make_grid(l_color.cpu(),nrow=8).permute(1,2,0).numpy())]},step=(epoch+1)*len(train_loader)-1)

        avg_loss = sum(val_losses)/len(val_losses)
        avg_dice = sum(val_dices)/len(val_dices)
        scheduler.step()
        print(f"[Epoch {epoch+1}] LR enc:{optimizer.param_groups[0]['lr']:.6f}, dec:{optimizer.param_groups[1]['lr']:.6f}")
        wandb.log({"valid_loss":avg_loss, "valid_dice_loss":avg_dice}, step=(epoch+1)*len(train_loader)-1)

        if avg_loss < best_valid:
            best_valid = avg_loss
            if best_path: os.remove(best_path)
            best_path = os.path.join(output_dir, f"best_ep{epoch+1:02}-loss{avg_loss:.4f}.pth")
            torch.save(model.state_dict(), best_path)

        # Final epoch per-class dice
        if epoch == args.epochs-1:
            dice_pc = (2*inters)/(preds_sum+labels_sum+1e-8)
            print("\nMean Dice per Category (Final Epoch):")
            print("| Class | Mean Dice |\n|---|---|")
            for i,name in enumerate(class_names): print(f"| {name} | {dice_pc[i]:.4f} |")
            table = wandb.Table(data=[[class_names[i], float(dice_pc[i])] for i in range(num_classes)], columns=["class","mean_dice"])
            wandb.log({"mean_dice_per_category":table})

    print("Training complete!")
    # Save final
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_ep{epoch+1:02}-loss{avg_loss:.4f}.pth"))
    wandb.finish()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
