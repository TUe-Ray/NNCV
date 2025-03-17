wandb login

python3 train_resnet34.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 10 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "resnet34+unet 64/10/0.001" \