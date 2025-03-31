wandb login

python3 SOTA_Dice_aug_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ConvNeXtUNet 16/30/1e-4 aug" \