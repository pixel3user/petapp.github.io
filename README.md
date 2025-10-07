# Stage 1 Dermatology Domain Adaptation - Complete Workflow

## 🎯 Overview

This workflow implements **Stage 1** of the two-step training plan for dermatology domain adaptation using Qwen2.5-VL. The goal is to teach the model visual-text alignment in the dermatology domain by training it to recognize skin conditions from images.

## 📊 Dataset Analysis Results

Based on comprehensive analysis of the four datasets in your Jupyter notebooks:

### Dataset Summary
| Dataset | Samples | Diseases | Avg Samples/Disease | Training Suitability |
|---------|---------|----------|-------------------|---------------------|
| **DermNet** | 19,559 | 23 | 850.4 | ✅ Excellent (all >200 samples) |
| **Fitzpatrick17k** | 16,577 | 114 | 145.4 | ✅ Excellent (all >50 samples) |
| **SCIN** | 3,061 | 210 | 14.6 | ⚠️ Mixed (many <20 samples) |
| **DDI** | 656 | 78 | 8.4 | ⚠️ Poor (many <20 samples) |

**Total**: ~43,853 samples across 612 unique conditions

### Key Insights from Analysis
- **DermNet** and **Fitzpatrick17k** are excellent for training (all diseases have sufficient samples)
- **SCIN** has 68.6% of diseases with insufficient samples (<20) that should be filtered out
- **DDI** has 66.7% of diseases with insufficient samples that should be filtered out
- **Recommended minimum**: 20 samples per disease for reliable training

## 🚀 Complete Workflow

### Quick Start (Recommended)
```bash
./quick_start_stage1.sh
```

**Features:**
- ✅ Uses Qwen2-VL-Finetune library (optimized for Qwen2-VL)
- ✅ Liger-Kernel support for memory efficiency
- ✅ Advanced LoRA features
- ✅ DeepSpeed integration
- ✅ Mixed-modality dataset support

### Step-by-Step Execution

#### Step 1: Data Preparation
```bash
python stage1_data_preparation.py
```
**Output**: `stage1_data/train.jsonl`, `stage1_data/val.jsonl`, `stage1_data/metadata.json`

#### Step 2: Training
```bash
./run_stage1_training.sh
```
**Output**: `output/stage1_dermatology_qwen2vl/` (trained model)

#### Step 3: Evaluation
```bash
python stage1_evaluation_per_dataset.py \
    --model_path output/stage1_dermatology_qwen2vl \
    --data_dir stage1_data \
    --output_dir evaluation_results_per_dataset
```
**Output**: `evaluation_results_per_dataset/` (metrics and visualizations for each dataset)

## 📁 Generated Files

### Core Scripts
- `stage1_data_preparation.py` - Data preparation pipeline
- `stage1_evaluation_per_dataset.py` - Per-dataset evaluation script
- `stage1_config.json` - Training configuration
- `run_stage1_training.sh` - Training shell script (uses Qwen2-VL-Finetune)
- `quick_start_stage1.sh` - Complete workflow script

### Documentation
- `STAGE1_WORKFLOW.md` - Detailed workflow documentation
- `README.md` - This summary document

### Data Files
- `stage1_data/train.jsonl` - Training data (LLaVA format)
- `stage1_data/val.jsonl` - Validation data
- `stage1_data/metadata.json` - Dataset metadata

### Model Files
- `output/stage1_dermatology/adapter_config.json` - LoRA configuration
- `output/stage1_dermatology/adapter_model.bin` - LoRA weights
- `output/stage1_dermatology/training_config.json` - Training configuration

### Evaluation Files
- `evaluation_results/evaluation_results_test.json` - Detailed metrics
- `evaluation_results/confusion_matrix_test.png` - Confusion matrix
- `evaluation_results/metrics_test.png` - Performance metrics
- `evaluation_results/classification_report_test.txt` - Classification report

## 🔧 Technical Configuration

### Model Settings
- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Library**: Qwen2-VL-Finetune (optimized for Qwen2-VL)
- **Frozen Components**: Vision encoder, LLM
- **Trainable Components**: Projection layer (merger) via LoRA
- **LoRA Configuration**: Rank 64, Alpha 64, Dropout 0.05
- **Memory Optimization**: Liger-Kernel support
- **Multi-GPU**: DeepSpeed integration

### Training Parameters
- **Epochs**: 3
- **Batch Size**: 4 per device
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW with cosine schedule
- **Mixed Precision**: BF16 (optimized for Qwen2-VL)
- **Gradient Checkpointing**: Enabled
- **DeepSpeed**: Zero2 configuration
- **Liger-Kernel**: Enabled for memory efficiency

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
- **Loss**: Should decrease from ~2.5 to ~0.5
- **Memory Usage**: ~20-25GB GPU memory
- **Training Time**: 3-5 hours on modern GPU

### Evaluation Metrics
- **Accuracy**: Target >80% on validation set
- **F1-Score**: Target >0.75 (weighted average)
- **Per-Class Performance**: Good performance on common diseases

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

### Common Issues
- **Out of memory**: Reduce batch size, enable gradient checkpointing
- **Poor convergence**: Check learning rate, verify data quality
- **Low accuracy**: Check data filtering, analyze error patterns
- **Training too slow**: Increase batch size, use multiple GPUs

### Monitoring
- **Training loss**: Should decrease steadily
- **Validation loss**: Should track training loss
- **GPU memory**: Monitor usage
- **Training speed**: Samples per second

## 📚 Key Features

### Data Preparation
- ✅ Comprehensive dataset analysis
- ✅ Quality filtering (minimum 20 samples per disease)
- ✅ Unified dataset creation
- ✅ LLaVA format conversion
- ✅ Train/validation splits

### Training
- ✅ Optimized for dermatology domain
- ✅ LoRA fine-tuning for efficiency
- ✅ Comprehensive logging
- ✅ Automatic checkpointing
- ✅ Mixed precision training

### Evaluation
- ✅ Comprehensive metrics (accuracy, precision, recall, F1)
- ✅ Confusion matrix analysis
- ✅ Error analysis
- ✅ Performance visualizations
- ✅ Detailed reporting

## 🎉 Ready to Use

The workflow is now complete and ready for execution. Simply run:

```bash
./quick_start_stage1.sh
```

This will automatically:
1. Prepare the training data from your four datasets
2. Train the Qwen2.5-VL model for dermatology domain adaptation
3. Evaluate the trained model
4. Generate comprehensive reports and visualizations

The entire process should take 3-5 hours on a modern GPU, depending on your hardware configuration.