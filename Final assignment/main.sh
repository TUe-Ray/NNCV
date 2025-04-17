wandb login

python3 train_Segformer_Finetuned_copy.py \
    --data-dir ./data/cityscapes \
    --batch-size 4 \
    --epochs 2 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "TEST_FT_SegForB5 1024 FT_mean 4/2/1e-4_1e-7 aug" \