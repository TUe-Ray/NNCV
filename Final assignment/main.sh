wandb login

python3 ATrain_Segformer_Finetuned.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 20 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Noaug FT_SegForB5 512Dice FT_mean 8/20/1e-4_1e-7 " \