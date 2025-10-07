# Qwen2-VL Dataset Setup Guide

## Quick Start:
```bash
python setup_qwen2vl.py
```

## Manual Steps:
1. Convert dataset: `python convert_to_qwen2vl.py`
2. Validate dataset: `python validate_qwen2vl.py`
3. Download URLs (if needed): `python download_url_images.py`

## Files Created:
- `qwen2vl_train.json` - Training dataset
- `qwen2vl_val.json` - Validation dataset
- `downloaded_images/` - Downloaded URL images

## Qwen2-VL Training:
```bash
# Example training command (adjust paths as needed)
python -m qwen2vl_finetune.train \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --train_data_path qwen2vl_train.json \
    --eval_data_path qwen2vl_val.json \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 500 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --remove_unused_columns False \
    --report_to none
```

## Dataset Format:
Each sample should have:
- `id`: Unique identifier
- `image`: Path to image file
- `conversations`: List of conversation turns

Example:
```json
{
  "id": "dermatology_000001",
  "image": "data/ddidiversedermatologyimages/000001.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat skin condition is shown in this image?"
    },
    {
      "from": "gpt",
      "value": "This image shows melanoma-in-situ."
    }
  ]
}
```

## Requirements:
- Python 3.8+
- pandas
- PIL (Pillow)
- requests
- json

## Troubleshooting:
1. **Image not found**: Check if image paths are correct
2. **URL download fails**: Check internet connection and URL validity
3. **Large images**: Images >10MB or >12M pixels will be flagged
4. **Format issues**: Ensure images are in supported formats (PNG, JPG, etc.)
