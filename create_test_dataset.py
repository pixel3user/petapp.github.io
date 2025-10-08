#!/usr/bin/env python3
"""
Create a small test dataset for quick fine-tuning validation
"""

import json
import random
import os

def create_test_dataset(input_file, output_file, num_examples=1000):
    """Create a smaller dataset for testing"""
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original dataset size: {len(data)} examples")
    
    # Randomly sample examples
    if len(data) > num_examples:
        test_data = random.sample(data, num_examples)
    else:
        test_data = data
    
    print(f"Creating test dataset with {len(test_data)} examples...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save test dataset
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Test dataset saved to {output_file}")
    return len(test_data)

def create_test_config():
    """Create a test configuration for quick training"""
    
    # Create test data directory
    os.makedirs("stage1_test_data", exist_ok=True)
    
    # Create test datasets
    train_size = create_test_dataset("stage1_data/train.json", "stage1_test_data/train.json", 1000)
    val_size = create_test_dataset("stage1_data/val.json", "stage1_test_data/val.json", 200)
    
    print(f"\n✅ Test dataset created:")
    print(f"  - Training examples: {train_size}")
    print(f"  - Validation examples: {val_size}")
    print(f"  - Expected training time: 30-60 minutes (vs 10-15 hours)")
    print(f"  - Expected steps: ~16 steps per epoch (vs 625 steps)")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    create_test_config()
