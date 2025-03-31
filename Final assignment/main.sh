wandb login

python3 train_Weight_Dice_ResUnet101_aug_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0003 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ResUnet101 WeiDice_LROnPla 16/30/3e-4 aug" \