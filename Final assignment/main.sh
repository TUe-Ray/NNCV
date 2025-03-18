wandb login

python3 train_ResUnet101_Changingscheduler.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 30 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "ResUnet101 64/50/0.001  ChangingScheduler" \