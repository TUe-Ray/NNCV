wandb login

python3 train_1.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 20 \
    --lr 0.005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet_1 lr 0.005 epoch20" \