wandb login

python3 train_Tversky_ResUnet101_aug_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ResUnet101 TverskyLoss_LROnPla_gamma2 16/30/5e-4 aug" \