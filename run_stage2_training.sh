#!/bin/bash

# Stage 2 Training Script for Dermatology Educational Alignment
# This script trains the model on a small but rich dataset with comprehensive educational content

set -e

# Configuration
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_PATH="stage2_data"
OUTPUT_DIR="stage2_output"
CONFIG_FILE="stage2_config.json"

# Check if Stage 1 model exists
STAGE1_MODEL="stage1_output/checkpoint-1000"
if [ ! -d "$STAGE1_MODEL" ]; then
    echo "⚠️  Stage 1 model not found at $STAGE1_MODEL"
    echo "Please run Stage 1 training first or update the path"
    exit 1
fi

# Check if Stage 2 data exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "⚠️  Stage 2 data not found at $DATASET_PATH"
    echo "Please run stage2_data_preparation.py first"
    exit 1
fi

echo "🚀 Starting Stage 2 Training - Educational Alignment"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "Stage 1 Model: $STAGE1_MODEL"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Training command
echo "📚 Training on educational dataset with comprehensive responses..."
echo ""

deepspeed Qwen2-VL-Finetune/src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger False \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --deepspeed Qwen2-VL-Finetune/scripts/zero2.json \
    --model_id $STAGE1_MODEL \
    --data_path $DATASET_PATH \
    --image_folder "../" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --report_to none \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --bf16 True \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --seed 42 \
    --run_name "stage2_educational_alignment" \
    --description "Stage 2: Educational alignment training with rich dataset"

echo ""
echo "✅ Stage 2 Training Complete!"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Evaluate the model on test datasets"
echo "2. Test the model's educational response quality"
echo "3. Fine-tune hyperparameters if needed"
echo "4. Deploy the final model"
echo ""
echo "The model should now provide comprehensive educational responses including:"
echo "- Detailed diagnosis and explanation"
echo "- Symptom descriptions and management"
echo "- Safety precautions and advice"
echo "- Educational content about conditions"
echo "- Clarifying questions for better assessment"
