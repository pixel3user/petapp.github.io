# Stage 2 Dermatology Educational Alignment Workflow

## Overview

This workflow implements **Stage 2** of the two-step training plan for dermatology domain adaptation using Qwen2.5-VL. The goal is to teach the model to provide comprehensive, educational responses about dermatological conditions with proper medical guidance.

## 🎯 Stage 2 Objectives

- **Format**: Image → Comprehensive Educational Response
- **Training**: Fine-tune LLM (unfreeze) while keeping vision encoder frozen
- **Goal**: Teach Qwen2.5-VL to provide educational, safe, and helpful responses
- **Learning**: "This is psoriasis, here are the symptoms, precautions, and what you should know"

## 📊 Stage 2 Dataset Characteristics

### **Small but Rich Dataset (2,000 samples)**
Each sample contains comprehensive information:

| Component | Description | Example |
|-----------|-------------|---------|
| **Diagnosis** | Clear identification of condition | "This appears to be psoriasis" |
| **Symptoms** | Detailed symptom description | "Common symptoms include red, scaly patches, itching" |
| **Precautions** | Safety advice and warnings | "Avoid triggers like stress, use gentle skincare" |
| **Education** | Educational explanation | "Psoriasis is a chronic autoimmune condition..." |
| **Questions** | Clarifying questions | "How long have you had these patches?" |
| **Disclaimer** | Medical safety disclaimer | "Please consult with a healthcare provider" |

### **Disease Coverage**
- **Top 50 diseases** from Stage 1 dataset
- **Proportional sampling** based on disease frequency
- **5-50 samples per disease** for balanced coverage
- **Comprehensive educational content** for each condition

## 🚀 Workflow Steps

### Step 1: Data Preparation

Run the Stage 2 data preparation pipeline:

```bash
python stage2_data_preparation.py
```

**What it does:**
- Loads Stage 1 dataset to understand disease distribution
- Creates educational samples with rich information
- Generates comprehensive conversations with all components
- Creates train/validation splits (80/20)
- Saves training data in LLaVA format

**Output:**
- `stage2_data/train.jsonl` - 1,600 training samples
- `stage2_data/val.jsonl` - 400 validation samples
- `stage2_data/metadata.json` - Dataset statistics

### Step 2: Training

Run Stage 2 training:

```bash
./run_stage2_training.sh
```

**Training Configuration:**
- **Model**: Stage 1 checkpoint (domain-adapted)
- **Epochs**: 3 (small dataset, more epochs)
- **Learning Rate**: 1e-5 (lower for fine-tuning)
- **Batch Size**: 4 (with gradient accumulation)
- **LoRA**: Enabled (rank=64, alpha=16)
- **Vision Encoder**: Frozen (from Stage 1)
- **LLM**: Unfrozen (for educational alignment)

**Key Differences from Stage 1:**
- **LLM unfrozen** to learn educational responses
- **Lower learning rate** for fine-tuning
- **More epochs** due to smaller dataset
- **Educational focus** rather than visual-text alignment

### Step 3: Evaluation

Run comprehensive evaluation:

```bash
python stage2_evaluation.py
```

**Evaluation Metrics:**
- **Component Coverage**: Diagnosis, symptoms, precautions, education, questions
- **Quality Score**: Comprehensiveness, educational value, helpfulness
- **Safety Score**: Medical disclaimers, professional care recommendations
- **Disease-Specific Score**: Accuracy and relevance of disease information
- **Overall Score**: Combined performance across all metrics

## 📈 Expected Outcomes

### **Stage 2 Model Capabilities**
After training, the model should be able to:

1. **Provide Clear Diagnoses**
   - "Based on the visual characteristics, this appears to be eczema"
   - "The clinical presentation suggests psoriasis"

2. **Explain Symptoms**
   - "Common symptoms include red, scaly patches, itching, and burning sensation"
   - "You may experience dry skin and joint pain"

3. **Give Safety Precautions**
   - "Important precautions: avoid triggers like stress, use gentle skincare"
   - "Please note: seek medical evaluation if symptoms worsen"

4. **Provide Education**
   - "Eczema is a chronic inflammatory skin condition that causes dry, itchy skin"
   - "Psoriasis is an autoimmune condition that causes rapid skin cell turnover"

5. **Ask Clarifying Questions**
   - "To better assist you, could you tell me how long you've had these patches?"
   - "Additional information that would be helpful: do you have a family history?"

6. **Include Medical Disclaimers**
   - "Please note: This information is for educational purposes only and should not replace professional medical advice"

## 🔄 Two-Stage Training Benefits

### **Stage 1 (Domain Adaptation)**
- **Focus**: Visual-text alignment in dermatology
- **Training**: Freeze vision encoder + LLM, train projection layer
- **Outcome**: Model recognizes skin conditions from images

### **Stage 2 (Educational Alignment)**
- **Focus**: Comprehensive educational responses
- **Training**: Freeze vision encoder, fine-tune LLM
- **Outcome**: Model provides safe, educational, and helpful responses

### **Combined Benefits**
- **Domain expertise** from Stage 1
- **Educational quality** from Stage 2
- **Medical safety** through proper disclaimers
- **Comprehensive responses** with all necessary components

## 📊 Dataset Comparison

| Aspect | Stage 1 | Stage 2 |
|--------|---------|---------|
| **Size** | 40,487 samples | 2,000 samples |
| **Focus** | Visual-text alignment | Educational responses |
| **Components** | Diagnosis only | Diagnosis + Symptoms + Precautions + Education + Questions |
| **Training** | Projection layer | LLM fine-tuning |
| **Goal** | Recognize conditions | Provide comprehensive guidance |

## 🎯 Success Metrics

### **Component Coverage (Target: >80%)**
- Diagnosis mentioned
- Symptoms described
- Precautions provided
- Educational content included
- Questions asked
- Medical disclaimer present

### **Quality Scores (Target: >0.7)**
- Comprehensiveness
- Educational value
- Helpfulness
- Safety compliance

### **Overall Performance (Target: >0.75)**
- Combined score across all metrics
- Balanced performance across components
- Consistent quality across diseases

## 🚀 Next Steps After Stage 2

1. **Comprehensive Evaluation**
   - Test on diverse skin conditions
   - Evaluate response quality and safety
   - Assess educational value

2. **Model Deployment**
   - Deploy for real-world testing
   - Monitor user interactions
   - Collect feedback for improvements

3. **Continuous Improvement**
   - Refine based on user feedback
   - Add new diseases and conditions
   - Improve response quality

## 📁 File Structure

```
stage2_data/
├── train.jsonl          # Training samples
├── val.jsonl            # Validation samples
└── metadata.json        # Dataset statistics

stage2_output/
├── checkpoint-500/      # Training checkpoints
├── checkpoint-1000/
└── ...

stage2_evaluation/
├── stage2_evaluation_results.json
└── stage2_evaluation_report.md
```

## 🔧 Configuration Files

- **`stage2_config.json`**: Training configuration
- **`run_stage2_training.sh`**: Training script
- **`stage2_evaluation.py`**: Evaluation script

## 📝 Notes

- **Small dataset**: 2,000 samples is sufficient for educational alignment
- **Rich content**: Each sample contains comprehensive information
- **Safety first**: All responses include medical disclaimers
- **Educational focus**: Emphasis on teaching and guidance
- **Professional quality**: Responses suitable for medical education

This two-stage approach ensures both domain expertise and educational quality, resulting in a model that can safely and effectively assist with dermatological education and guidance.
