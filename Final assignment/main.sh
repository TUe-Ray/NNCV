wandb login

python3 test_load_model.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 2 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "load model" \