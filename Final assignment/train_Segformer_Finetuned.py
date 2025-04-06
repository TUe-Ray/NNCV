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

Feel free to customize the script as needed for your use case.
"""
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
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, LinearLR, CyclicLR
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
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

    train_transform = Compose([
        ToImage(),
        RandomHorizontalFlip(p=0.5),
        Resize((512, 512)),
        #RandomCrop((256, 256), pad_if_needed=True),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ToDtype(torch.float32, scale=True),
        RandomRotation(degrees=30),
        GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        Normalize(mean=processor.image_mean, std=processor.image_std)
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Validation: 保持最簡單的處理
    valid_transform = Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=processor.image_mean, std=processor.image_std)
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])




    # Load the dataset and make a split for training and validation
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

    
   # 載入預訓練的 config，然後修改 num_labels
    config = SegformerConfig.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        num_channels=3,
        num_labels = 19,
        upsample_ratio = 1
    )

    # 使用修改後的 config 初始化模型
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        config=config,
        ignore_mismatched_sizes=True  # 如果 label 數跟原本不同，這個參數是關鍵！
    ).to(device)
    # Define the loss function
    # 使用 SMP 內建的 DiceLoss（針對多分類任務）
    # 注意：此處使用 mode='multiclass'，並可設定 ignore_index 來忽略 void 類別
    
    
    #criterion = smp.losses.DiceLoss(mode='multiclass', log_loss = True, ignore_index=255)
    criterion = smp.losses.DiceLoss(mode='multiclass',  ignore_index=255)
    dice_loss_fn = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)# 新增：Dice Loss


        # Define the optimizer
        # 分離 encoder 與其他部分的參數
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    # 定義優化器，給 encoder 使用較小的學習率（例如：0.1 * args.lr）
    optimizer = AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},
        {'params': decoder_params, 'lr': args.lr}
    ])
    # scheduler = CyclicLR(
    #     optimizer,
    #     base_lr=1e-5,
    #     max_lr=1e-3,
    #     step_size_up=len(train_dataloader)*5,  # 1 個 epoch warmup
    #     step_size_down=len(train_dataloader)*10,            # 不設的話，up/down cycle 對稱
    #     mode='triangular2',
    #     cycle_momentum=False,           # 如果你用的是 Adam，要設 False
    # )
    
    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    
    
    
    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs.logits  # [B, C, H, W]
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
            dice_losses = []  # 新增：用來累積 Dice loss
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                outputs = model(images)
                logits = outputs.logits  # [B, C, H, W]

                logits = torch.nn.functional.interpolate(
                    logits, size=labels.shape[1:], mode='bilinear', align_corners=False
                )
                loss = criterion(logits, labels)
                losses.append(loss.item())

                # 計算 Dice Loss
                dice_loss_val = dice_loss_fn(logits, labels)
                dice_losses.append(dice_loss_val.item())

                if i == 0:
                    predictions = logits.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)

            
            
            
            valid_loss = sum(losses) / len(losses)
            valid_dice_loss = sum(dice_losses) / len(dice_losses)  # 平均 Dice loss
            scheduler.step(valid_loss)
            wandb.log({
                "valid_loss": valid_loss,
                "valid_dice_loss": valid_dice_loss,  # 新增：Dice loss log
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
