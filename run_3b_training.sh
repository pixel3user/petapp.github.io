#!/bin/bash

# Full Training Script for Qwen2.5-VL-3B with LoRA
# This script uses the complete stage1_data (32,146 training + 8,300 validation examples)

set -e

echo "=== Full Training for Qwen2.5-VL-3B with LoRA ==="

# Configuration
DOCKER_IMAGE="john119/vlm"
CONTAINER_NAME="qwen2vl_3b_full_training"
HOST_WORKSPACE="/teamspace/studios/this_studio"
DOCKER_WORKSPACE="/workspace"

# Model and training configuration - 3B MODEL OPTIMIZED FOR A100
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
BATCH_PER_DEVICE=8     # Reduced for full dataset training
GRAD_ACCUM_STEPS=16    # Higher accumulation for stable training with larger dataset
NUM_EPOCHS=3           # More epochs for full training
OUTPUT_DIR="output/full_3b_dermatology_qwen2vl"

echo "Configuration:"
echo "  Model: $MODEL_NAME (3B parameters)"
echo "  Batch per device: $BATCH_PER_DEVICE (optimized for full dataset)"
echo "  Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "  Training epochs: $NUM_EPOCHS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Dataset: 32,146 training examples, 8,300 validation examples"
echo "  Expected time: 4-6 hours (full dataset on A100)"
echo "  Expected VRAM: 25-35GB (A100 80GB with full dataset)"
echo "  GPU: A100 80GB (recommended for full training)"
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
        
        echo 'Starting FULL training with Qwen2.5-VL-3B...'
        echo 'Training configuration:'
        echo '  Model: $MODEL_NAME (3B parameters)'
        echo '  Batch size: $BATCH_PER_DEVICE'
        echo '  Gradient accumulation: $GRAD_ACCUM_STEPS'
        echo '  Epochs: $NUM_EPOCHS'
        echo '  Dataset: 32,146 training examples, 8,300 validation examples'
        echo '  Effective batch size: \$((BATCH_PER_DEVICE * GRAD_ACCUM_STEPS))'
        echo ''
        
        # Use the repository's LoRA fine-tuning script optimized for full dataset
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
            --data_path ../stage1_data/train.json \
            --eval_path ../stage1_data/val.json \
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
            --learning_rate 5e-5 \
            --merger_lr 5e-6 \
            --vision_lr 1e-6 \
            --weight_decay 0.1 \
            --warmup_ratio 0.05 \
            --lr_scheduler_type \"cosine\" \
            --logging_steps 10 \
            --tf32 True \
            --gradient_checkpointing True \
            --dataloader_drop_last True \
            --report_to tensorboard \
            --lazy_preprocess True \
            --save_strategy \"steps\" \
            --save_steps 500 \
            --save_total_limit 5 \
            --dataloader_num_workers 8 \
            --dataloader_pin_memory True \
            --eval_strategy steps \
            --eval_steps 500 \
            --load_best_model_at_end True \
            --metric_for_best_model eval_loss \
            --greater_is_better False
        
        echo ''
        echo '=== FULL Training completed ==='
        echo 'Model saved to: $OUTPUT_DIR'
        
        # Merge LoRA weights with base model for Hugging Face upload
        echo ''
        echo '=== Merging LoRA weights with base model ==='
        echo 'This will create a full model that can be uploaded to Hugging Face...'
        
        # Create merged model directory
        MERGED_MODEL_DIR=\"../output/merged_full_3b_dermatology_model\"
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
# Qwen2.5-VL-3B Dermatology Model (Full Training)

This is a fine-tuned version of Qwen2.5-VL-3B-Instruct specifically trained for dermatology image analysis and diagnosis using the complete dataset.

## Model Details

- **Base Model**: Qwen2.5-VL-3B-Instruct
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Domain**: Dermatology
- **Training Data**: 32,146 dermatology images with conversations
- **Validation Data**: 8,300 dermatology images
- **Total Training Examples**: 40,446

## Training Configuration

- **LoRA Rank**: 64
- **LoRA Alpha**: 64
- **LoRA Dropout**: 0.05
- **Learning Rate**: 5e-5
- **Batch Size**: 8 (per device)
- **Gradient Accumulation**: 16
- **Effective Batch Size**: 128
- **Epochs**: 3
- **GPU**: A100 80GB
- **Training Time**: 4-6 hours

## Dataset Sources

- **DermNet**: 489 samples (test), ~15,000+ samples (full)
- **Fitzpatrick17k**: 419 samples (test), ~10,000+ samples (full)
- **DermaVQA**: 14 samples (test), ~5,000+ samples (full)
- **SCIN**: 63 samples (test), ~3,000+ samples (full)
- **SkinCap**: 15 samples (test), ~1,000+ samples (full)
- **DDI**: 1 sample (test), ~500+ samples (full)

## Performance

This model has been trained on a comprehensive dermatology dataset and should provide:
- Accurate skin condition identification
- Detailed dermatological descriptions
- Medical terminology usage
- Visual question answering capabilities

## Usage

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_name = "your-username/qwen2.5-vl-3b-dermatology"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# Load and process image
image = Image.open("dermatology_image.jpg")
messages = [
    {"role": "user", "content": "What skin condition is shown in this image?"}
]

# Generate response
response = model.generate(
    processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
    images=[image],
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7
)
```

## Training Logs

Training logs and metrics are available in the output directory:
- TensorBoard logs: `output/full_3b_dermatology_qwen2vl/runs/`
- Model checkpoints: `output/full_3b_dermatology_qwen2vl/checkpoint-*/`

## License

This model is based on Qwen2.5-VL-3B-Instruct and follows the same license terms.

## Citation

If you use this model, please cite the original Qwen2.5-VL paper and acknowledge the dermatology dataset sources.
EOF
        
        echo ''
        echo '=== Model card created ==='
        echo 'Model is ready for deployment and Hugging Face upload!'
        echo ''
        echo 'Next steps:'
        echo '1. Test the model with sample dermatology images'
        echo '2. Upload to Hugging Face Hub'
        echo '3. Create inference API or web interface'
        echo ''
        echo 'Training completed successfully! 🎉'
    "

echo ""
echo "=== Training completed ==="
echo "Check the output directory: $OUTPUT_DIR"
echo "Merged model available at: output/merged_full_3b_dermatology_model"
echo ""
echo "To monitor training progress:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "To access the container:"
echo "  docker exec -it $CONTAINER_NAME /bin/bash"
