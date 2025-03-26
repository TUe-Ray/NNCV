wandb login

python3 train_Dice_ResUnet101_aug_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ResUnet101 log_Dice_warmup up 16/30/0.0005 aug" \