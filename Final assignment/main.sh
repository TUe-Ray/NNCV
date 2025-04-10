wandb login

python3 train_Weight_Dice_Segformer_FT.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "FT_SegForB5 WeightDice_512 FT_mean 16/30/1e-4_1e-6 aug" \