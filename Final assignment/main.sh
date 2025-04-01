wandb login

python3 train_Dice_Segformer_aug.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Segformer finetuned mean 16/30/5e-4 aug" \