wandb login

python3 train_Dice_ResUnet101_aug_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 40 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id " RU101 logDice 16/40/1e-4_1e-6 Noaug" \