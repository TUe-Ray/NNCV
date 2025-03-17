wandb login

python3 train_1.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 5 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet1 512 epoch 5" \