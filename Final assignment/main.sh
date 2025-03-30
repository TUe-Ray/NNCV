wandb login

python3 train_Dice_ResUnet34_aug_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ResUnet34 Dice_LROnPla 16/30/5e-4 aug" \