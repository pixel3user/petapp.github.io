# Comprehensive Qwen2-VL Setup Script
import pandas as pd
import json
import os
import requests
from PIL import Image
from pathlib import Path
import time

def setup_qwen2vl_dataset(csv_path="stage1_data/unified_dataset.csv"):
    """Complete setup for Qwen2-VL finetuning"""
    
    print("="*60)
    print("QWEN2-VL DATASET SETUP")
    print("="*60)
    
    # Step 1: Load and analyze dataset
    print("\nStep 1: Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Step 2: Check image accessibility
    print("\nStep 2: Checking image accessibility...")
    # Handle NaN values in image_path column
    df = df.dropna(subset=['image_path'])  # Remove rows with NaN image_path
    print(f"After removing NaN image paths: {len(df)} samples")
    
    url_samples = df[df['image_path'].str.startswith('http')]
    local_samples = df[~df['image_path'].str.startswith('http')]
    
    print(f"URL images: {len(url_samples)}")
    print(f"Local images: {len(local_samples)}")
    
    # Step 3: Download URL images if needed
    if len(url_samples) > 0:
        print(f"\nStep 3: Downloading URL images...")
        df = download_url_images(df, url_samples)
    
    # Step 4: Convert to Qwen2-VL format
    print(f"\nStep 4: Converting to Qwen2-VL format...")
    qwen_data = convert_to_qwen2vl_format(df)
    
    # Step 5: Create train/val split
    print(f"\nStep 5: Creating train/validation split...")
    train_data, val_data = create_train_val_split(qwen_data, df)
    
    # Step 6: Validate final dataset
    print(f"\nStep 6: Validating final dataset...")
    validate_final_dataset(train_data, val_data)
    
    print(f"\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print(f"Files created:")
    print(f"- qwen2vl_train.json ({len(train_data)} samples)")
    print(f"- qwen2vl_val.json ({len(val_data)} samples)")
    print(f"- downloaded_images/ (if URLs were downloaded)")
    
    return train_data, val_data

def download_url_images(df, url_samples):
    """Download URL images to local storage"""
    
    output_dir = "downloaded_images"
    Path(output_dir).mkdir(exist_ok=True)
    
    downloaded_count = 0
    failed_count = 0
    
    for idx, row in url_samples.iterrows():
        url = row['image_path']
        dataset = row.get('dataset', 'unknown')
        disease = row.get('disease', 'unknown')
        
        # Create filename (sanitize for filesystem)
        safe_disease = "".join(c for c in str(disease) if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{dataset}_{safe_disease}_{idx}.jpg"
        local_path = os.path.join(output_dir, filename)
        
        try:
            # Download image
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save image
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            # Update the dataframe
            df.at[idx, 'image_path'] = local_path
            downloaded_count += 1
            
            if downloaded_count % 100 == 0:
                print(f"Downloaded {downloaded_count} images...")
                
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            failed_count += 1
    
    print(f"Download complete: {downloaded_count} success, {failed_count} failed")
    return df

def convert_to_qwen2vl_format(df):
    """Convert DataFrame to Qwen2-VL format"""
    
    qwen_data = []
    
    for idx, row in df.iterrows():
        try:
            entry = {
                "id": f"dermatology_{idx:06d}",
                "image": row['image_path'],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\nWhat skin condition is shown in this image?"
                    },
                    {
                        "from": "gpt", 
                        "value": f"This image shows {row.get('disease', 'unknown condition')}."
                    }
                ]
            }
            qwen_data.append(entry)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    return qwen_data

def create_train_val_split(qwen_data, df):
    """Create train/validation split based on original split column"""
    
    train_data = []
    val_data = []
    
    for i, item in enumerate(qwen_data):
        original_split = df.iloc[i]['split']
        if original_split == 'train':
            train_data.append(item)
        else:
            val_data.append(item)
    
    # Save files
    with open('qwen2vl_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('qwen2vl_val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    return train_data, val_data

def validate_final_dataset(train_data, val_data):
    """Validate the final dataset"""
    
    print(f"Validating train set ({len(train_data)} samples)...")
    train_valid = validate_samples(train_data)
    
    print(f"Validating validation set ({len(val_data)} samples)...")
    val_valid = validate_samples(val_data)
    
    print(f"\nFinal validation results:")
    print(f"Train set: {train_valid}/{len(train_data)} valid")
    print(f"Validation set: {val_valid}/{len(val_data)} valid")
    
    return train_valid, val_valid

def validate_samples(samples):
    """Validate a list of samples"""
    
    valid_count = 0
    
    for item in samples:
        try:
            image_path = item['image']
            if os.path.exists(image_path):
                img = Image.open(image_path)
                img.verify()
                valid_count += 1
        except:
            pass
    
    return valid_count

# Run the setup
if __name__ == "__main__":
    train_data, val_data = setup_qwen2vl_dataset()
