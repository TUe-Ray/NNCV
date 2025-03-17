wandb login

python3 train_1.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 10 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet1 512 lr0.001 batch 16 epoch 10 /3C normalization" \