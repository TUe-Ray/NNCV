wandb login

python3 train_ResUnet101_sqscheduler.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 50 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ResUnet101 64/50/0.001  SequentialScheduler" \