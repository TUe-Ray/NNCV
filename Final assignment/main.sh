wandb login

python3 ATrain_CEDiceSegformer_Finetuned.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 20 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id " FT_SegForB5 7CE3Dice 512FT_mean 8/20/1e-4_1e-7 Noaug" \