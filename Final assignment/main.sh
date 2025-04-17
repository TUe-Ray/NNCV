wandb login

python3 train_Segformer_Finetuned_copy.py \
    --data-dir ./data/cityscapes \
    --batch-size 2 \
    --epochs 1 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "TEST_audocast" \