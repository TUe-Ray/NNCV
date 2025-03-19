wandb login

python3 train_ResUnet101_augumentation.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 40 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ResUnet101 64/40/0.001 Aug-2" \