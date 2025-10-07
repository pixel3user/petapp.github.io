#!/bin/bash

# Stage 1 Training Script using Qwen2-VL-Finetune Library
# =======================================================
# 
# This script uses the official Qwen2-VL-Finetune library for Stage 1 training:
# - Image → Disease Label (short captions)
# - Freeze vision encoder + LLM, train projection layer only
# - Optimized for Qwen2-VL with Liger-Kernel support
#
# Prerequisites:
# 1. Run stage1_data_preparation.py first to prepare training data
# 2. Install Qwen2-VL-Finetune requirements
# 3. Set up wandb account for logging (optional)

set -e  # Exit on any error

# Configuration
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_DIR="stage1_data"
OUTPUT_DIR="output/stage1_dermatology_qwen2vl"
CONFIG_FILE="stage1_config_qwen2vl.json"
WANDB_PROJECT="dermatology-stage1-qwen2vl"

# Training parameters
NUM_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=2e-4
LORA_RANK=64

# GPU settings
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Check if Qwen2-VL-Finetune is available
if [ ! -d "Qwen2-VL-Finetune" ]; then
    echo "❌ Qwen2-VL-Finetune directory not found!"
    echo "Please ensure the Qwen2-VL-Finetune library is available"
    exit 1
fi

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory $DATA_DIR not found!"
    echo "Please run stage1_data_preparation.py first to prepare training data"
    exit 1
fi

if [ ! -f "$DATA_DIR/train.jsonl" ] || [ ! -f "$DATA_DIR/val.jsonl" ]; then
    echo "❌ Training files not found in $DATA_DIR!"
    echo "Please run stage1_data_preparation.py first to prepare training data"
    exit 1
fi

# Convert JSONL to JSON format (required by Qwen2-VL-Finetune)
echo "Converting JSONL to JSON format..."
python -c "
import json
import sys

def convert_jsonl_to_json(jsonl_file, json_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f'Converted {len(data)} samples from {jsonl_file} to {json_file}')

convert_jsonl_to_json('$DATA_DIR/train.jsonl', '$DATA_DIR/train.json')
convert_jsonl_to_json('$DATA_DIR/val.jsonl', '$DATA_DIR/val.json')
"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if wandb is available
if command -v wandb &> /dev/null; then
    echo "✓ Wandb found - logging will be enabled"
    WANDB_ENABLED="true"
else
    echo "⚠️  Wandb not found - logging will be disabled"
    WANDB_ENABLED="false"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU by default
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="$WANDB_PROJECT"

# Print configuration
echo "=========================================="
echo "Stage 1 Dermatology Domain Adaptation"
echo "Using Qwen2-VL-Finetune Library"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "LoRA rank: $LORA_RANK"
echo "Wandb project: $WANDB_PROJECT"
echo "=========================================="

# Check GPU memory
echo "Checking GPU memory..."
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

# Run training using Qwen2-VL-Finetune
echo "Starting training with Qwen2-VL-Finetune..."

cd Qwen2-VL-Finetune

# Calculate gradient accumulation steps
GRAD_ACCUM_STEPS=$((64 / (BATCH_SIZE * NUM_GPUS)))
if [ $GRAD_ACCUM_STEPS -lt 1 ]; then
    GRAD_ACCUM_STEPS=1
fi

echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"

# Run training with DeepSpeed
deepspeed src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_RANK \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id "$MODEL_NAME" \
    --data_path "../$DATA_DIR/train.json" \
    --eval_dataset_path "../$DATA_DIR/val.json" \
    --image_folder "../" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "../$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --dataloader_num_workers 4 \
    --eval_strategy "steps" \
    --eval_steps 500

cd ..

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    
    # Display final results
    echo "=========================================="
    echo "Training Results Summary"
    echo "=========================================="
    echo "Output directory: $OUTPUT_DIR"
    echo "Model files:"
    ls -la "$OUTPUT_DIR"
    
    # Check if LoRA weights need to be merged
    if [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
        echo ""
        echo "📝 Note: LoRA weights were saved separately."
        echo "To merge LoRA weights with base model, run:"
        echo "cd Qwen2-VL-Finetune && bash scripts/merge_lora.sh"
        echo "Update the paths in scripts/merge_lora.sh first!"
    fi
    
    echo ""
    echo "Next steps:"
    echo "1. Evaluate the model on test data"
    echo "2. Proceed to Stage 2 (Instruction/Educational Alignment)"
    echo "3. Fine-tune hyperparameters if needed"
    
else
    echo "❌ Training failed!"
    echo "Check the logs above for error details"
    exit 1
fi

echo "=========================================="
echo "Stage 1 Training Complete!"
echo "=========================================="
