# Qwen2.5-VL Dermatology Fine-tuning Guide

This guide explains how to fine-tune Qwen2.5-VL-7B-Instruct on multiple dermatology datasets using the Qwen2-VL-Finetune framework.

## Dataset Overview

The unified dataset combines 5 dermatology datasets:

1. **DermaVQA** (0 samples) - Medical Q&A pairs (images not found)
2. **Fitzpatrick17k** (1,658 samples) - Disease classification with skin tone diversity
3. **SCIN** (5,032 samples) - Comprehensive skin condition cases with symptoms
4. **SkinCap** (1,000 samples) - Skin condition images with synthetic captions
5. **DDI** (656 samples) - Diverse dermatology images with disease labels

**Total**: 8,346 samples (6,677 train, 1,669 validation)
**Image Availability**: 80.1% (6,688/8,346 images accessible)

## Dataset Format

The unified dataset follows the Qwen2-VL-Finetune format:

```json
{
  "conversations": [
    {"from": "human", "value": "What skin condition is shown in this image?"},
    {"from": "gpt", "value": "This image shows [condition]. Based on the visual characteristics..."}
  ],
  "image": "path/to/image.jpg",
  "source": "dataset_name"
}
```

## Prerequisites

1. **Install Dependencies**:
```bash
cd Qwen2-VL-Finetune
pip install -r requirements.txt
```

2. **Install Additional Packages**:
```bash
pip install datasets pandas pillow
```

3. **Verify GPU Memory**: Ensure you have at least 24GB GPU memory for 7B model training.

## Training Configuration

### Key Parameters:
- **Model**: Qwen/Qwen2.5-VL-7B-Instruct
- **LoRA Rank**: 64 (efficient fine-tuning)
- **Learning Rate**: 2e-4
- **Batch Size**: 4 per device (64 global with gradient accumulation)
- **Epochs**: 3
- **Image Resolution**: 672x672 (Qwen2.5-VL requirement)

### Memory Optimization:
- **DeepSpeed ZeRO-3**: Reduces memory usage
- **Gradient Checkpointing**: Trades compute for memory
- **LoRA**: Only trains adapter weights
- **Frozen Components**: Vision tower, LLM, and merger frozen initially

## Training Steps

### 1. Prepare Dataset
The unified dataset is already created in `unified_dataset/`:
- `train.json`: 6,677 training samples
- `val.json`: 1,669 validation samples

### 2. Start Training
```bash
cd Qwen2-VL-Finetune
bash ../train_dermatology_model.sh
```

### 3. Monitor Training
- **TensorBoard**: `tensorboard --logdir output/qwen2.5-vl-dermatology`
- **Logs**: Check console output for loss and metrics
- **Checkpoints**: Saved every 500 steps in `output/qwen2.5-vl-dermatology/`

### 4. Merge LoRA Weights (Optional)
After training, merge LoRA weights for inference:
```bash
python src/merge_lora_weights.py \
    --base_model Qwen/Qwen2.5-VL-7B-Instruct \
    --lora_model output/qwen2.5-vl-dermatology \
    --output_dir output/qwen2.5-vl-dermatology/merged
```

## Expected Training Time

- **Hardware**: Single A100 80GB GPU
- **Estimated Time**: 8-12 hours for 3 epochs
- **Steps**: ~1,250 steps per epoch (6,677 samples / 64 batch size)

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce `BATCH_PER_DEVICE` to 2 or 1
   - Increase `GRAD_ACCUM_STEPS` accordingly
   - Use `zero3_offload.json` instead of `zero3.json`

2. **Image Loading Errors**:
   - Check image paths in dataset
   - Ensure all images are accessible
   - Some Fitzpatrick17k images may be missing (URLs)

3. **Slow Training**:
   - Enable `tf32` for faster training
   - Use `flash_attention_2` (default)
   - Increase `dataloader_num_workers`

### Memory Usage Optimization:
```bash
# For lower memory GPUs (16-24GB)
BATCH_PER_DEVICE=2
GRAD_ACCUM_STEPS=32
# Use zero3_offload.json for CPU offloading
```

## Evaluation

The model will be evaluated on validation set every 500 steps. Key metrics:
- **Loss**: Should decrease over time
- **Perplexity**: Lower is better
- **BLEU/ROUGE**: For response quality (if implemented)

## Next Steps

1. **Test the Model**: Use the fine-tuned model for dermatology Q&A
2. **Further Fine-tuning**: Adjust hyperparameters based on results
3. **Domain Adaptation**: Fine-tune on specific dermatology subdomains
4. **Evaluation**: Test on held-out dermatology datasets

## Dataset Sources

- **DermaVQA**: Medical Q&A dataset
- **Fitzpatrick17k**: Skin tone classification dataset
- **SCIN**: Skin condition identification dataset
- **SkinCap**: Skin condition captioning dataset
- **DDI**: Diverse dermatology images dataset

## Notes

- The dataset includes synthetic Q&A pairs for datasets without text
- Some images may be missing (especially Fitzpatrick17k URLs)
- The model is trained for medical consultation, not diagnosis
- Always recommend professional medical consultation in responses


