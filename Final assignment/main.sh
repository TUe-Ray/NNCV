wandb login

python3 train_Segformer_Finetuned.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 30 \
    --lr 0.000005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Finetuned_Segformerb5 0.5_mean 16/30/5e-6 aug" \