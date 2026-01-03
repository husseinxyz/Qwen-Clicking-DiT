#!/bin/bash
# Training script for click prediction with DiT action head
# Frozen VLM backbone + trainable DiT head

export PYTHONPATH=src:$PYTHONPATH

# Model
MODEL_ID="Qwen/Qwen2.5-VL-2B-Instruct"

# Dataset
DATASET="HelloKKMe/grounding_dataset"

# Output
OUTPUT_DIR="outputs/click_dit"

# Resolution (controls visual tokens)
# 256*28*28=200704 min, 512*28*28=401408 max
IMAGE_MIN_PIXELS=200704
IMAGE_MAX_PIXELS=401408

# Training
BATCH_SIZE=4
GRAD_ACCUM=4  # Effective batch = 16
LR=1e-4
EPOCHS=3

python -m src.train.train_click \
    --model_id $MODEL_ID \
    --data_path $DATASET \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --image_min_pixels $IMAGE_MIN_PIXELS \
    --image_max_pixels $IMAGE_MAX_PIXELS \
    --freeze_merger False \
    --mlp_head_dim 512 \
    --mlp_head_dropout 0.1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --disable_flash_attn2 False \
    --report_to "none"
