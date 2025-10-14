#!/bin/bash

# Quick Test Training Script for Qwen2.5-VL-3B with LoRA
# This script uses a small dataset (1000 examples) for quick validation with 3B model

set -e

echo "=== Quick Test Training for Qwen2.5-VL-3B with LoRA ==="

# Configuration
DOCKER_IMAGE="john119/vlm"
CONTAINER_NAME="qwen2vl_3b_test_training"
HOST_WORKSPACE="/teamspace/studios/this_studio"
DOCKER_WORKSPACE="/workspace"

# Model and training configuration - 3B MODEL OPTIMIZED FOR A100
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
BATCH_PER_DEVICE=16   # Conservative for 3B model on A100
GRAD_ACCUM_STEPS=8    # Higher accumulation for stable training
NUM_EPOCHS=1
OUTPUT_DIR="output/test_3b_dermatology_qwen2vl"

echo "Configuration:"
echo "  Model: $MODEL_NAME (3B parameters - 2-3x faster than 7B)"
echo "  Batch per device: $BATCH_PER_DEVICE (conservative for 3B model)"
echo "  Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "  Training epochs: $NUM_EPOCHS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Dataset: 1000 training examples, 200 validation examples"
echo "  Expected time: 10-15 minutes (3B model on A100)"
echo "  Expected VRAM: 20-30GB (comfortable fit on A100 80GB)"
echo "  GPU: A100 80GB (efficient utilization)"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if the Docker image exists, if not pull it
if ! docker image inspect $DOCKER_IMAGE > /dev/null 2>&1; then
    echo "Pulling Docker image: $DOCKER_IMAGE"
    docker pull $DOCKER_IMAGE
fi

# Stop and remove existing container if it exists
if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping and removing existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME > /dev/null 2>&1 || true
    docker rm $CONTAINER_NAME > /dev/null 2>&1 || true
fi

# Create output directory if it doesn't exist
mkdir -p "$HOST_WORKSPACE/$OUTPUT_DIR"

echo "Starting Docker container with GPU support..."
echo "Host workspace: $HOST_WORKSPACE"
echo "Docker workspace: $DOCKER_WORKSPACE"
echo ""

# Run the Docker container with GPU support and volume mounting
docker run --gpus all \
    --name $CONTAINER_NAME \
    --ipc=host \
    -v "$HOST_WORKSPACE:$DOCKER_WORKSPACE" \
    -w "$DOCKER_WORKSPACE" \
    $DOCKER_IMAGE \
    /bin/bash -c "
        echo '=== Inside Docker Container ==='
        echo 'Python version:' \$(python --version)
        echo 'CUDA available:' \$(python -c 'import torch; print(torch.cuda.is_available())')
        echo 'GPU count:' \$(python -c 'import torch; print(torch.cuda.device_count())')
        echo ''
        
        # Activate the conda environment
        echo 'Activating conda environment: train'
        source /opt/conda/etc/profile.d/conda.sh
        conda activate train
        
        # Navigate to Qwen2-VL-Finetune directory
        cd Qwen2-VL-Finetune
        
        pip install -r requirements.txt -f https://download.pytorch.org/whl/cu128
        pip install qwen-vl-utils
        pip install flash-attn --no-build-isolation

        # Set PYTHONPATH
        export PYTHONPATH=src:\$PYTHONPATH
        
        echo 'Starting TEST training with Qwen2.5-VL-3B...'
        echo 'Training configuration:'
        echo '  Model: $MODEL_NAME (3B parameters)'
        echo '  Batch size: $BATCH_PER_DEVICE'
        echo '  Gradient accumulation: $GRAD_ACCUM_STEPS'
        echo '  Epochs: $NUM_EPOCHS'
        echo '  Dataset: 1000 examples'
        echo ''
        
        # Use the repository's LoRA fine-tuning script with conservative A100 settings
        deepspeed src/train/train_sft.py \
            --use_liger True \
            --lora_enable True \
            --use_dora False \
            --lora_namespan_exclude \"['lm_head', 'embed_tokens']\" \
            --lora_rank 64 \
            --lora_alpha 64 \
            --lora_dropout 0.05 \
            --num_lora_modules -1 \
            --deepspeed scripts/zero2.json \
            --model_id $MODEL_NAME \
            --data_path ../stage1_test_data/train.json \
            --eval_path ../stage1_test_data/val.json \
            --image_folder ../ \
            --remove_unused_columns False \
            --freeze_vision_tower False \
            --freeze_llm True \
            --freeze_merger False \
            --bf16 True \
            --fp16 False \
            --disable_flash_attn2 False \
            --output_dir ../$OUTPUT_DIR \
            --num_train_epochs $NUM_EPOCHS \
            --per_device_train_batch_size $BATCH_PER_DEVICE \
            --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
            --image_min_pixels \$((256 * 28 * 28)) \
            --image_max_pixels \$((896 * 28 * 28)) \
            --learning_rate 1e-4 \
            --merger_lr 1e-5 \
            --vision_lr 2e-6 \
            --weight_decay 0.1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type \"cosine\" \
            --logging_steps 1 \
            --tf32 True \
            --gradient_checkpointing True \
            --dataloader_drop_last True \
            --report_to tensorboard \
            --lazy_preprocess True \
            --save_strategy \"steps\" \
            --save_steps 5 \
            --save_total_limit 3 \
            --dataloader_num_workers 8 \
            --dataloader_pin_memory True \
            --eval_strategy steps \
            --eval_steps 3
        
        echo ''
        echo '=== TEST Training completed ==='
        echo 'Model saved to: $OUTPUT_DIR'
        
        # Merge LoRA weights with base model for Hugging Face upload
        echo ''
        echo '=== Merging LoRA weights with base model ==='
        echo 'This will create a full model that can be uploaded to Hugging Face...'
        
        # Create merged model directory
        MERGED_MODEL_DIR=\"../output/merged_3b_dermatology_model\"
        mkdir -p \$MERGED_MODEL_DIR
        
        # Use the repository's merge script
        python src/merge_lora_weights.py \
            --model-path ../$OUTPUT_DIR \
            --model-base $MODEL_NAME \
            --save-model-path \$MERGED_MODEL_DIR \
            --safe-serialization
        
        echo ''
        echo '=== Model merging completed ==='
        echo 'Merged model saved to: \$MERGED_MODEL_DIR'
        echo 'This model is ready for Hugging Face upload!'
        
        # Create model card for Hugging Face
        echo ''
        echo '=== Creating model card ==='
        cat > \$MERGED_MODEL_DIR/README.md << 'EOF'
# Qwen2.5-VL-3B Dermatology Model

This is a fine-tuned version of Qwen2.5-VL-3B-Instruct specifically trained for dermatology image analysis and diagnosis.

## Model Details

- **Base Model**: Qwen2.5-VL-3B-Instruct
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Domain**: Dermatology
- **Training Data**: 1,000 dermatology images with conversations
- **Validation Data**: 200 dermatology images

## Training Configuration

- **LoRA Rank**: 64
- **LoRA Alpha**: 64
- **LoRA Dropout**: 0.05
- **Learning Rate**: 1e-4
- **Batch Size**: 16
- **Epochs**: 1
- **Gradient Accumulation**: 8
- **GPU**: A100 80GB (efficient utilization)

## Usage

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

# Load the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(\"./merged_3b_dermatology_model\")
processor = AutoProcessor.from_pretrained(\"./merged_3b_dermatology_model\")

# Load and process image
image = Image.open(\"path_to_dermatology_image.jpg\")
inputs = processor(
    text=\"<image>\\nWhat skin condition is shown in this image?\",
    images=image,
    return_tensors=\"pt\"
)

# Generate response
outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Datasets

This model was trained on a combination of dermatology datasets:
- DermNet (via kagglehub)
- Fitzpatrick17k
- DDI (Diverse Dermatology Images)
- SCIN (Skin Cancer Image Network)
- SkinCap

## Limitations

- This is a test model trained on a small dataset (1,000 examples)
- For production use, consider training on a larger, more diverse dataset
- Always consult with medical professionals for actual diagnosis

## License

This model inherits the license from the base Qwen2.5-VL-3B-Instruct model.
EOF
        
        echo 'Model card created: \$MERGED_MODEL_DIR/README.md'
        
        # Create requirements.txt for the model
        cat > \$MERGED_MODEL_DIR/requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.37.0
pillow>=9.0.0
accelerate>=0.20.0
EOF
        
        echo 'Requirements file created: \$MERGED_MODEL_DIR/requirements.txt'
        
        # Display final summary
        echo ''
        echo '=== FINAL SUMMARY ==='
        echo '✅ LoRA training completed'
        echo '✅ Model weights merged'
        echo '✅ Model card created'
        echo '✅ Requirements file created'
        echo ''
        echo '📁 Output directories:'
        echo '  - LoRA weights: ../$OUTPUT_DIR'
        echo '  - Merged model: \$MERGED_MODEL_DIR'
        echo ''
        echo '🚀 Ready for Hugging Face upload!'
        echo '   Use: huggingface-cli upload <repo-name> \$MERGED_MODEL_DIR'
    "

echo ""
echo "=== Test training completed ==="
echo "Container name: $CONTAINER_NAME"
echo "To access the container: docker exec -it $CONTAINER_NAME /bin/bash"
echo "To view logs: docker logs $CONTAINER_NAME"
echo "To stop container: docker stop $CONTAINER_NAME"
echo "To remove container: docker rm $CONTAINER_NAME"
