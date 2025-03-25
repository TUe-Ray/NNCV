wandb login

python3 train_DiceCE_ResUnet101_aug_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 10 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "DiceCE ResUnet101 512 16/10/0.0005 aug" \