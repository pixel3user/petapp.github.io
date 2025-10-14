# Qwen2.5-VL-3B Dermatology Model Training Guide

This guide explains how to train a 3B parameter dermatology model and upload it to Hugging Face Hub.

## 🚀 Quick Start

### 1. Run Training
```bash
./run_3b_test_training.sh
```

### 2. Upload to Hugging Face (after training completes)
```bash
./upload_to_huggingface.sh
```

## 📋 What the Modified Script Does

### **Training Phase:**
- ✅ Uses **Qwen2.5-VL-3B-Instruct** (3B parameters - faster than 7B)
- ✅ Trains on **1,000 dermatology images** (quick test)
- ✅ Uses **LoRA fine-tuning** (efficient training)
- ✅ Saves LoRA weights to `output/test_3b_dermatology_qwen2vl/`

### **Post-Training Phase:**
- ✅ **Merges LoRA weights** with base model
- ✅ Creates **full model** ready for Hugging Face
- ✅ Generates **model card** (README.md)
- ✅ Creates **requirements.txt**
- ✅ Saves merged model to `output/merged_3b_dermatology_model/`

## 📊 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | Qwen2.5-VL-3B-Instruct | 3B parameters (2-3x faster than 7B) |
| **Batch Size** | 32 | Larger batch size possible with 3B model |
| **LoRA Rank** | 64 | Good balance of efficiency and performance |
| **Learning Rate** | 1e-4 | Standard learning rate |
| **Epochs** | 1 | Quick test training |
| **Expected Time** | 15-30 minutes | vs 10-15 hours for 7B model |
| **Expected VRAM** | 15-25GB | vs 35-45GB for 7B model |

## 📁 Output Structure

After training, you'll have:

```
output/
├── test_3b_dermatology_qwen2vl/          # LoRA weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training logs...
└── merged_3b_dermatology_model/          # Full model for HF
    ├── config.json
    ├── model.safetensors
    ├── tokenizer files...
    ├── README.md                         # Model card
    └── requirements.txt
```

## 🔧 Prerequisites

### **System Requirements:**
- Docker with GPU support
- NVIDIA GPU with 15-25GB VRAM
- Internet connection for model download

### **Docker Image:**
- Uses `john119/vlm` image
- Automatically pulls if not present

## 📝 Training Data

The script uses your verified test data:
- **Train**: 1,000 samples from `stage1_test_data/train.json`
- **Validation**: 200 samples from `stage1_test_data/val.json`
- **Datasets**: DermNet, Fitzpatrick, DDI, SCIN, SkinCap
- **All image paths verified** ✅

## 🚀 Hugging Face Upload

### **Automatic Upload:**
```bash
./upload_to_huggingface.sh
```

### **Manual Upload:**
```bash
# Install huggingface_hub
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Upload model
huggingface-cli upload your-username/qwen2.5-vl-3b-dermatology output/merged_3b_dermatology_model --repo-type model
```

## 🧪 Testing Your Model

### **Local Testing:**
```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

# Load your trained model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("./output/merged_3b_dermatology_model")
processor = AutoProcessor.from_pretrained("./output/merged_3b_dermatology_model")

# Test with a dermatology image
image = Image.open("path_to_dermatology_image.jpg")
inputs = processor(
    text="<image>\nWhat skin condition is shown in this image?",
    images=image,
    return_tensors="pt"
)

# Generate response
outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### **Hugging Face Spaces:**
1. Go to your model page on Hugging Face
2. Click "Create Space"
3. Use the Gradio template for image classification
4. Your model will be available for public testing!

## 📊 Expected Performance

### **Training Metrics:**
- **Loss**: Should decrease steadily
- **Validation**: Monitored every 3 steps
- **Memory**: 15-25GB VRAM usage
- **Time**: 15-30 minutes total

### **Model Performance:**
- **Speed**: 2-3x faster inference than 7B model
- **Quality**: Good for dermatology tasks (test dataset)
- **Size**: ~6GB (vs ~14GB for 7B model)

## 🔍 Monitoring Training

### **View Training Logs:**
```bash
# View container logs
docker logs qwen2vl_3b_test_training

# Access container
docker exec -it qwen2vl_3b_test_training /bin/bash
```

### **TensorBoard:**
```bash
# Start TensorBoard (if available)
tensorboard --logdir output/test_3b_dermatology_qwen2vl/runs
```

## 🛠️ Troubleshooting

### **Common Issues:**

1. **Out of Memory:**
   - Reduce `BATCH_PER_DEVICE` to 16 or 8
   - Increase `GRAD_ACCUM_STEPS` to 8

2. **Docker Issues:**
   - Ensure Docker has GPU support
   - Check NVIDIA Docker runtime

3. **Model Upload Fails:**
   - Check Hugging Face login: `huggingface-cli whoami`
   - Verify repository name format

### **Performance Tuning:**
- **Faster Training**: Reduce `NUM_EPOCHS` to 0.5
- **Better Quality**: Increase `NUM_EPOCHS` to 2-3
- **Memory Optimization**: Use gradient checkpointing (already enabled)

## 📈 Next Steps

### **After Training:**
1. ✅ Test model locally
2. ✅ Upload to Hugging Face
3. ✅ Create Hugging Face Space
4. ✅ Share with community

### **For Production:**
1. Train on full dataset (not just test data)
2. Use more epochs (3-5)
3. Fine-tune hyperparameters
4. Add more diverse dermatology data

## 🎯 Key Benefits of 3B Model

- **⚡ Speed**: 2-3x faster than 7B model
- **💾 Memory**: 50% less VRAM required
- **📦 Size**: Smaller model for deployment
- **🔄 Iteration**: Faster experimentation
- **💰 Cost**: Lower compute costs

Your 3B dermatology model will be ready for Hugging Face in about 30 minutes!



