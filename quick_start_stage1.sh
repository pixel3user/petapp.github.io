#!/bin/bash

# Quick Start Script for Stage 1 Dermatology Domain Adaptation
# Using Qwen2-VL-Finetune Library
# ============================================================
#
# This script provides a complete workflow for Stage 1 training:
# 1. Data preparation
# 2. Training execution (using Qwen2-VL-Finetune)
# 3. Model evaluation
#
# Usage: ./quick_start_stage1_qwen2vl.sh

set -e  # Exit on any error

echo "=========================================="
echo "Stage 1 Dermatology Domain Adaptation"
echo "Using Qwen2-VL-Finetune Library"
echo "Quick Start Workflow"
echo "=========================================="

# Configuration
DATA_DIR="stage1_data"
OUTPUT_DIR="output/stage1_dermatology_qwen2vl"
EVAL_DIR="evaluation_results_qwen2vl"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

# Check required packages
echo "Checking required packages..."
python -c "import torch, transformers, peft, PIL, pandas, numpy, sklearn, matplotlib, seaborn" 2>/dev/null || {
    echo "❌ Missing required packages. Please install:"
    echo "pip install torch transformers peft pillow pandas numpy scikit-learn matplotlib seaborn wandb"
    exit 1
}

# Check Qwen2-VL-Finetune
if [ ! -d "Qwen2-VL-Finetune" ]; then
    echo "❌ Qwen2-VL-Finetune directory not found!"
    echo "Please ensure the Qwen2-VL-Finetune library is available"
    exit 1
fi

# Check DeepSpeed
if ! command -v deepspeed &> /dev/null; then
    echo "❌ DeepSpeed not found. Please install DeepSpeed:"
    echo "pip install deepspeed"
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No GPU detected. Training will be slow on CPU."
fi

echo "✓ Prerequisites check complete"

# Step 1: Data Preparation
echo ""
echo "=========================================="
echo "Step 1: Data Preparation"
echo "=========================================="

if [ ! -d "$DATA_DIR" ] || [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "Preparing training data..."
    python stage1_data_preparation.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Data preparation completed"
    else
        echo "❌ Data preparation failed"
        exit 1
    fi
else
    echo "✓ Training data already exists"
fi

# Convert JSONL to JSON format (required by Qwen2-VL-Finetune)
echo "Converting data format for Qwen2-VL-Finetune..."
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

# Step 2: Training
echo ""
echo "=========================================="
echo "Step 2: Model Training"
echo "=========================================="

if [ ! -d "$OUTPUT_DIR" ] || [ ! -f "$OUTPUT_DIR/adapter_config.json" ]; then
    echo "Starting model training with Qwen2-VL-Finetune..."
    echo "This may take several hours depending on your hardware..."
    
    # Run training
    ./run_stage1_training_qwen2vl.sh
    
    if [ $? -eq 0 ]; then
        echo "✅ Training completed"
    else
        echo "❌ Training failed"
        exit 1
    fi
else
    echo "✓ Trained model already exists"
fi

# Step 3: Evaluation
echo ""
echo "=========================================="
echo "Step 3: Model Evaluation"
echo "=========================================="

echo "Evaluating trained model on each dataset separately..."
python stage1_evaluation_per_dataset.py \
    --model_path "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$EVAL_DIR"

if [ $? -eq 0 ]; then
    echo "✅ Evaluation completed"
else
    echo "❌ Evaluation failed"
    exit 1
fi

# Final Summary
echo ""
echo "=========================================="
echo "Stage 1 Workflow Complete!"
echo "=========================================="
echo ""
echo "📁 Output Files:"
echo "  Model: $OUTPUT_DIR"
echo "  Evaluation: $EVAL_DIR"
echo "  Training Data: $DATA_DIR"
echo ""
echo "📊 Key Files:"
echo "  - $OUTPUT_DIR/adapter_config.json (LoRA config)"
echo "  - $OUTPUT_DIR/adapter_model.bin (LoRA weights)"
echo "  - $EVAL_DIR/evaluation_results_test.json (metrics)"
echo "  - $EVAL_DIR/confusion_matrix_test.png (visualization)"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review evaluation results"
echo "  2. Analyze confusion matrix and error patterns"
echo "  3. Consider Stage 2 training if results are satisfactory"
echo "  4. Fine-tune hyperparameters if needed"
echo ""
echo "📚 Documentation:"
echo "  - STAGE1_WORKFLOW.md (detailed workflow)"
echo "  - stage1_config_qwen2vl.json (training configuration)"
echo "  - Qwen2-VL-Finetune/README.md (library documentation)"
echo ""
echo "🔧 Advanced Options:"
echo "  - To merge LoRA weights: cd Qwen2-VL-Finetune && bash scripts/merge_lora.sh"
echo "  - To use different DeepSpeed config: modify scripts/zero2.json"
echo "  - To enable vision LoRA: set --vision_lora True"
echo ""
echo "=========================================="
echo "Workflow completed successfully!"
echo "=========================================="
