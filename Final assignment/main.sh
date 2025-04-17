wandb login

python3 train_Segformer_Finetuned_copy.py \
    --data-dir ./data/cityscapes \
    --batch-size 2 \
    --epochs 20 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "FT_SegForB5 1024WDice FT_mean 2/20/1e-4_1e-7 aug" \