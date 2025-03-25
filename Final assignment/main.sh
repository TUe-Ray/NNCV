wandb login

python3 train_Dice_ResUnet101_aug_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 2 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "test" \