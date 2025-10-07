# Stage 1 Dermatology Domain Adaptation Workflow

## Overview

This workflow implements **Stage 1** of the two-step training plan for dermatology domain adaptation using Qwen2.5-VL. The goal is to teach the model visual-text alignment in the dermatology domain by training it to recognize skin conditions from images.

## 🎯 Stage 1 Objectives

- **Format**: Image → Disease Label (short captions)
- **Training**: Freeze vision encoder + LLM, train projection layer only
- **Goal**: Teach Qwen2.5-VL visual-text alignment in dermatology domain
- **Learning**: "This lesion looks like psoriasis vs melanoma vs acne"

## 📊 Dataset Analysis Summary

Based on comprehensive analysis of the four datasets:

| Dataset | Samples | Diseases | Avg Samples/Disease | Training Suitability |
|---------|---------|----------|-------------------|---------------------|
| **DermNet** | 19,559 | 23 | 850.4 | ✅ Excellent (all >200 samples) |
| **Fitzpatrick17k** | 16,577 | 114 | 145.4 | ✅ Excellent (all >50 samples) |
| **SCIN** | 3,061 | 210 | 14.6 | ⚠️ Mixed (many <20 samples) |
| **DDI** | 656 | 78 | 8.4 | ⚠️ Poor (many <20 samples) |

**Total**: ~43,853 samples across 612 unique conditions

### Key Insights:
- **DermNet** and **Fitzpatrick17k** are excellent for training (all diseases have sufficient samples)
- **SCIN** and **DDI** have many diseases with insufficient samples (<20) that should be filtered out
- **Recommended minimum**: 20 samples per disease for reliable training

## 🚀 Workflow Steps

### Step 1: Data Preparation

Run the data preparation pipeline to create unified training data:

```bash
python stage1_data_preparation.py
```

**What it does:**
- Loads all four datasets (DDI, Fitzpatrick17k, SCIN, DermNet)
- Analyzes dataset quality and provides recommendations
- Filters diseases with insufficient samples (<20 samples)
- Creates unified dataset in LLaVA format
- Generates train/validation splits
- Saves training data as JSONL files

**Output:**
- `stage1_data/train.jsonl` - Training data
- `stage1_data/val.jsonl` - Validation data
- `stage1_data/metadata.json` - Dataset metadata

### Step 2: Training Configuration

Review and adjust training parameters in `stage1_config.json`:

```json
{
  "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
  "freeze_vision_encoder": true,
  "freeze_llm": true,
  "freeze_merger": true,
  "use_lora": true,
  "lora_rank": 64,
  "num_train_epochs": 3,
  "per_device_train_batch_size": 4,
  "learning_rate": 2e-4,
  "image_size": 672
}
```

**Key Parameters:**
- **Freeze settings**: Vision encoder, LLM, and merger are frozen (only projection layer trains)
- **LoRA**: Rank 64 for efficient fine-tuning
- **Batch size**: 4 per device (adjust based on GPU memory)
- **Learning rate**: 2e-4 (conservative for domain adaptation)

### Step 3: Training Execution

Run the training script:

```bash
# Option 1: Use the shell script (recommended)
./run_stage1_training.sh

# Option 2: Run Python script directly
python stage1_training.py \
    --config stage1_config.json \
    --data_dir stage1_data \
    --output_dir output/stage1_dermatology \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --lora_rank 64
```

**Training Features:**
- Automatic GPU detection and setup
- Wandb logging (optional)
- Checkpoint saving every 500 steps
- Validation every 500 steps
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16)

### Step 4: Model Evaluation

Evaluate the trained model:

```bash
python stage1_evaluation.py \
    --model_path output/stage1_dermatology \
    --data_dir stage1_data \
    --output_dir evaluation_results
```

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Classification report
- Error analysis
- Performance visualizations

## 📁 File Structure

```
├── stage1_data_preparation.py    # Data preparation pipeline
├── stage1_training.py            # Training script
├── stage1_evaluation.py          # Evaluation script
├── stage1_config.json           # Training configuration
├── run_stage1_training.sh       # Training shell script
├── stage1_data/                 # Prepared training data
│   ├── train.jsonl
│   ├── val.jsonl
│   └── metadata.json
├── output/stage1_dermatology/   # Trained model
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_config.json
└── evaluation_results/          # Evaluation results
    ├── evaluation_results_test.json
    ├── confusion_matrix_test.png
    └── metrics_test.png
```

## 🔧 Technical Details

### Model Architecture
- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Frozen Components**: Vision encoder, LLM, merger
- **Trainable Components**: Projection layer only (via LoRA)
- **LoRA Configuration**: Rank 64, Alpha 64, Dropout 0.05

### Training Strategy
- **Domain Adaptation**: Image → Disease Label mapping
- **Format**: LLaVA-style conversations
- **Loss**: Cross-entropy on disease labels
- **Optimizer**: AdamW with cosine learning rate schedule
- **Regularization**: Weight decay 0.1, warmup ratio 0.03

### Data Format
```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat skin condition is shown in this image?"
    },
    {
      "from": "gpt",
      "value": "This image shows psoriasis."
    }
  ]
}
```

## 📈 Expected Results

### Training Metrics
- **Loss**: Should decrease steadily from ~2.5 to ~0.5
- **Learning Rate**: Cosine schedule from 2e-4 to 0
- **Memory Usage**: ~20-25GB GPU memory with batch size 4

### Evaluation Metrics
- **Accuracy**: Target >80% on validation set
- **F1-Score**: Target >0.75 (weighted average)
- **Per-Class Performance**: Good performance on common diseases (>100 samples)

### Common Issues
- **Low accuracy on rare diseases**: Expected due to insufficient training data
- **Overfitting**: Monitor validation loss, reduce learning rate if needed
- **Memory issues**: Reduce batch size or enable gradient checkpointing

## 🎯 Success Criteria

Stage 1 is successful when:
1. **Validation accuracy >80%** on filtered dataset
2. **Model can distinguish** between common skin conditions
3. **Loss converges** without overfitting
4. **No catastrophic forgetting** of base model capabilities

## 🔄 Next Steps

After successful Stage 1 completion:
1. **Evaluate model performance** on test data
2. **Analyze error patterns** and identify weaknesses
3. **Prepare for Stage 2**: Instruction/Educational Alignment
4. **Collect additional data** for poorly performing classes if needed

## 🛠️ Troubleshooting

### Common Issues and Solutions

**Issue**: Out of memory errors
- **Solution**: Reduce batch size, enable gradient checkpointing, use FP16

**Issue**: Poor convergence
- **Solution**: Check learning rate, verify data quality, increase warmup

**Issue**: Low accuracy on validation
- **Solution**: Check data filtering, verify image paths, analyze error patterns

**Issue**: Training too slow
- **Solution**: Increase batch size, use multiple GPUs, optimize data loading

### Monitoring Training

**Key Metrics to Watch:**
- Training loss (should decrease steadily)
- Validation loss (should track training loss)
- Learning rate (cosine schedule)
- GPU memory usage
- Training speed (samples/second)

**Red Flags:**
- Validation loss increasing while training loss decreases (overfitting)
- Loss not decreasing after several epochs (learning rate too low)
- Memory errors (batch size too large)
- NaN losses (learning rate too high)

## 📚 References

- [Qwen2.5-VL Documentation](https://github.com/QwenLM/Qwen2-VL)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [LLaVA Training](https://github.com/haotian-liu/LLaVA)
- [Dermatology Datasets](https://github.com/derrickburns/dermatology-datasets)

## 🤝 Contributing

To improve this workflow:
1. Test with different hyperparameters
2. Add support for additional datasets
3. Implement advanced evaluation metrics
4. Optimize training efficiency
5. Add support for multi-GPU training

---

**Note**: This workflow is designed for Stage 1 domain adaptation. For Stage 2 (Instruction/Educational Alignment), a different approach will be needed with richer, more detailed training data.
