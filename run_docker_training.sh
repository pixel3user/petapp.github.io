#!/bin/bash

# Docker-based Stage 1 Training Script for Qwen2.5-VL-7B with LoRA
# This script uses the john119/vlm Docker image to avoid GLIBC compatibility issues

set -e

echo "=== Docker-based Stage 1 Training for Qwen2.5-VL-7B with LoRA ==="

# Configuration
DOCKER_IMAGE="john119/vlm"
CONTAINER_NAME="qwen2vl_training"
HOST_WORKSPACE="/teamspace/studios/this_studio"
DOCKER_WORKSPACE="/workspace"

# Model and training configuration
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
BATCH_PER_DEVICE=24
GRAD_ACCUM_STEPS=4
NUM_EPOCHS=3
OUTPUT_DIR="output/stage1_dermatology_qwen2vl"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Batch per device: $BATCH_PER_DEVICE"
echo "  Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "  Training epochs: $NUM_EPOCHS"
echo "  Output directory: $OUTPUT_DIR"
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
docker run --gpus all -i \
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
        
        echo 'Starting training with Qwen2.5-VL-7B...'
        echo 'Training configuration:'
        echo '  Model: $MODEL_NAME'
        echo '  Batch size: $BATCH_PER_DEVICE'
        echo '  Gradient accumulation: $GRAD_ACCUM_STEPS'
        echo '  Epochs: $NUM_EPOCHS'
        echo ''
        
        # Use the repository's LoRA fine-tuning script
        bash scripts/finetune_lora.sh
        
        echo ''
        echo '=== Training completed ==='
        echo 'Model saved to: $OUTPUT_DIR'
    "

echo ""
echo "=== Docker training completed ==="
echo "Container name: $CONTAINER_NAME"
echo "To access the container: docker exec -it $CONTAINER_NAME /bin/bash"
echo "To view logs: docker logs $CONTAINER_NAME"
echo "To stop container: docker stop $CONTAINER_NAME"
echo "To remove container: docker rm $CONTAINER_NAME"
