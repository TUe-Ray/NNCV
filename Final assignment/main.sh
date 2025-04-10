wandb login

python3 train_Segformer_Finetuned.py \
    --data-dir ./data/cityscapes \
    --batch-size 4 \
    --epochs 30 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "FinTun_Segforb5 1024 FT_mean 8/30/1e-4_1e-6 aug" \