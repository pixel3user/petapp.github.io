#!/usr/bin/env python3
"""
Script to download the SkinCAP dataset from Hugging Face
This dataset requires authentication to access.
"""

import os
from datasets import load_dataset

def download_skincap_dataset():
    """Download the SkinCAP dataset to the data folder"""
    
    # Set the output directory
    output_dir = "/teamspace/studios/this_studio/data/skincap"
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading SkinCAP dataset...")
    print("Note: This dataset requires Hugging Face authentication.")
    
    try:
        # Load the dataset
        ds = load_dataset("joshuachou/SkinCAP")
        
        print(f"Dataset downloaded successfully!")
        print(f"Dataset info:")
        print(f"- Train split: {len(ds['train'])} samples")
        if 'validation' in ds:
            print(f"- Validation split: {len(ds['validation'])} samples")
        if 'test' in ds:
            print(f"- Test split: {len(ds['test'])} samples")
        
        # Save the dataset to the specified directory
        print(f"Saving dataset to {output_dir}...")
        ds.save_to_disk(output_dir)
        
        print("Dataset saved successfully!")
        
        # Print some sample information
        print("\nSample from the dataset:")
        print(ds['train'][0])
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTo access this dataset, you need to:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'Read' access")
        print("3. Run: huggingface-cli login")
        print("4. Enter your token when prompted")
        print("5. Then run this script again")
        return False
    
    return True

if __name__ == "__main__":
    download_skincap_dataset()