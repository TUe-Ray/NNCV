wandb login

python3 train_Segformer_Finetuned_copy.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 20 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id " flipCropRotate FT_SegForB5 512FT_mean 8/20/1e-3_1e-7 Noaug" \