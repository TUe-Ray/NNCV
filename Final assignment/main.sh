wandb login

python3 train_ResUnet101_augumentation_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ResUnet101 512 16/30/0.0001 aug" \