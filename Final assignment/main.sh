wandb login

python3 train_Segformer_Finetuned.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.00001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Finetuned_Segformerb5 finetuned_mean 16/30/1e-5 aug" \