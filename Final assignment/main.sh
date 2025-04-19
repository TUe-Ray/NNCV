wandb login

python3 train_Segformer_Finetuned_copy.py \
    --data-dir ./data/cityscapes \
    --batch-size 2 \
    --epochs 10 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id " NoAug FT_SegForB5_forRobutnesss 1024  2/10/1e-4_1e-7" \