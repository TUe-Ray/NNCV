wandb login

python3 Segformer.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Segformer-b2-finetuned-cityscapes-1024-1024 16/30/1e-4 aug" \