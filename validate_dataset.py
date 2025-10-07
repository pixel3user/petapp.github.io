# Dataset Validation Script
import pandas as pd
import os
from PIL import Image

def validate_dataset(csv_path):
    """Validate that all images in the dataset can be loaded"""
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Handle NaN values in image_path column
    df = df.dropna(subset=['image_path'])
    print(f"After removing NaN image paths: {len(df)} samples")
    print(f"Validating images...")
    
    valid_count = 0
    invalid_count = 0
    invalid_samples = []
    
    for idx, row in df.iterrows():
        path = row['image_path']
        
        try:
            if os.path.exists(path):
                img = Image.open(path)
                img.verify()  # Verify it's a valid image
                valid_count += 1
            else:
                invalid_count += 1
                invalid_samples.append((idx, path, "File not found"))
                
        except Exception as e:
            invalid_count += 1
            invalid_samples.append((idx, path, str(e)))
        
        if (idx + 1) % 1000 == 0:
            print(f"Validated {idx + 1} images...")
    
    print(f"\nValidation complete!")
    print(f"Valid images: {valid_count}")
    print(f"Invalid images: {invalid_count}")
    
    if invalid_samples:
        print(f"\nFirst 10 invalid samples:")
        for idx, path, error in invalid_samples[:10]:
            print(f"  Row {idx}: {path} - {error}")
    
    return valid_count, invalid_count, invalid_samples

# Usage example:
if __name__ == "__main__":
    validate_dataset('stage1_data/unified_dataset.csv')
