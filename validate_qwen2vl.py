# Qwen2-VL Dataset Validation Script
import pandas as pd
import json
import os
from PIL import Image

def validate_qwen2vl_dataset(json_path):
    """Validate Qwen2-VL dataset format and image accessibility"""
    
    # Load JSON dataset
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Validating {len(data)} samples...")
    
    valid_count = 0
    invalid_count = 0
    invalid_samples = []
    
    for item in data:
        try:
            # Check required fields
            if 'id' not in item or 'image' not in item or 'conversations' not in item:
                invalid_count += 1
                invalid_samples.append((item.get('id', 'unknown'), "Missing required fields"))
                continue
            
            # Check image accessibility
            image_path = item['image']
            if os.path.exists(image_path):
                # Check image format and size
                img = Image.open(image_path)
                img.verify()
                
                # Check file size
                file_size = os.path.getsize(image_path)
                if file_size > 10 * 1024 * 1024:  # 10MB
                    invalid_count += 1
                    invalid_samples.append((item['id'], f"Image too large: {file_size/1024/1024:.1f}MB"))
                    continue
                
                # Check pixel count
                img = Image.open(image_path)
                pixel_count = img.size[0] * img.size[1]
                if pixel_count > 12 * 1000 * 1000:  # 12M pixels
                    invalid_count += 1
                    invalid_samples.append((item['id'], f"Too many pixels: {pixel_count:,}"))
                    continue
                
                valid_count += 1
            else:
                invalid_count += 1
                invalid_samples.append((item['id'], f"Image not found: {image_path}"))
                
        except Exception as e:
            invalid_count += 1
            invalid_samples.append((item.get('id', 'unknown'), f"Error: {str(e)}"))
    
    print(f"\nValidation complete!")
    print(f"Valid samples: {valid_count}")
    print(f"Invalid samples: {invalid_count}")
    
    if invalid_samples:
        print(f"\nFirst 10 invalid samples:")
        for sample_id, error in invalid_samples[:10]:
            print(f"  {sample_id}: {error}")
    
    return valid_count, invalid_count, invalid_samples

# Usage example:
if __name__ == "__main__":
    # Validate the converted dataset
    validate_qwen2vl_dataset('qwen2vl_dataset.json')
