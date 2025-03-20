wandb login

python3 train_ResUnet101_augumentation_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 10 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ResUnet101 512 32/10/0.0005 aug" \