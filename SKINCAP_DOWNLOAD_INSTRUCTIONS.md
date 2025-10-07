# SkinCAP Dataset Download Instructions

## Prerequisites
The SkinCAP dataset requires Hugging Face authentication to download.

## Steps to Download:

1. **Get a Hugging Face Token:**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Select "Read" access
   - Copy the generated token

2. **Login to Hugging Face CLI:**
   ```bash
   huggingface-cli login
   ```
   - Paste your token when prompted
   - Choose "Y" for adding token as git credential

3. **Run the download script:**
   ```bash
   python download_skincap.py
   ```

## Alternative: Manual Download
If you prefer to download manually, you can also use:
```python
from datasets import load_dataset
ds = load_dataset("joshuachou/SkinCAP")
ds.save_to_disk("/teamspace/studios/this_studio/data/skincap")
```

## Dataset Information
- **Name:** SkinCAP
- **Author:** joshuachou
- **Description:** A dermatology dataset for skin condition analysis
- **Location:** Will be saved to `/teamspace/studios/this_studio/data/skincap/`
