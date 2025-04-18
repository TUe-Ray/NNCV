wandb login

python3 train_Segformer_Raw.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 20 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id " Raw_SegForB5 512FT_mean 8/20/1e-4_1e-7 aug" \