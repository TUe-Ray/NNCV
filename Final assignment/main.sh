wandb login

python3 train_TverskyLoss_ResUnet101_aug_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ResUnet34 TverskyLoss_LROnPla_gamma1.5 16/30/5e-4 aug" \