# Qwen2-VL Dataset Conversion Script
import pandas as pd
import json
import os
from pathlib import Path

def convert_to_qwen2vl_format(csv_path, output_path="qwen2vl_dataset.json"):
    """Convert CSV dataset to Qwen2-VL format"""
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Handle NaN values in image_path column
    df = df.dropna(subset=['image_path'])
    print(f"After removing NaN image paths: {len(df)} samples")
    print(f"Converting to Qwen2-VL format...")
    
    qwen_data = []
    
    for idx, row in df.iterrows():
        try:
            # Create Qwen2-VL format entry
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
        
        if (idx + 1) % 1000 == 0:
            print(f"Converted {idx + 1} samples...")
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(qwen_data, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Saved {len(qwen_data)} samples to {output_path}")
    
    return qwen_data

def create_train_val_split(qwen_data, train_ratio=0.8):
    """Split dataset into train and validation sets"""
    
    # Filter by split if available
    train_data = [item for item in qwen_data if item['id'].endswith('_train')]
    val_data = [item for item in qwen_data if item['id'].endswith('_val')]
    
    if not train_data and not val_data:
        # If no split info, create random split
        import random
        random.shuffle(qwen_data)
        split_idx = int(len(qwen_data) * train_ratio)
        train_data = qwen_data[:split_idx]
        val_data = qwen_data[split_idx:]
    
    # Save train and val files
    with open('qwen2vl_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('qwen2vl_val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"\nSplit complete!")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data

# Usage example:
if __name__ == "__main__":
    # Convert the unified dataset
    qwen_data = convert_to_qwen2vl_format('stage1_data/unified_dataset.csv')
    
    # Create train/val split
    train_data, val_data = create_train_val_split(qwen_data)
