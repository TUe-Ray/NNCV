wandb login

python3 train_SotaUnet_Dice_aug_512.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "convnext_base_21k 16/30/1e-4 aug" \