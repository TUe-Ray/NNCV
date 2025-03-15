wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 75 \
    --lr 0.005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-training" \