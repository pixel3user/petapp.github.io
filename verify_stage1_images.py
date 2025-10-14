#!/usr/bin/env python3
"""
Stage1 Data Image Path Verification Script
==========================================

This script verifies that all image paths in stage1_data/train.json and stage1_data/val.json
actually exist and are accessible.

Usage:
    python verify_stage1_images.py
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image

def verify_image_paths(json_file_path):
    """Verify that all image paths in a JSON file exist and are valid images"""
    
    print(f"\n{'='*60}")
    print(f"Verifying: {json_file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(json_file_path):
        print(f"❌ ERROR: File not found: {json_file_path}")
        return 0, 0, []
    
    # Load JSON data
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ ERROR: Failed to load JSON file: {e}")
        return 0, 0, []
    
    print(f"📊 Total samples: {len(data)}")
    
    valid_count = 0
    invalid_count = 0
    invalid_samples = []
    
    for idx, item in enumerate(data):
        try:
            # Check required fields
            if 'image' not in item:
                invalid_count += 1
                invalid_samples.append((idx, item.get('id', f'sample_{idx}'), "Missing 'image' field"))
                continue
            
            image_path = item['image']
            
            # Check if file exists
            if not os.path.exists(image_path):
                invalid_count += 1
                invalid_samples.append((idx, item.get('id', f'sample_{idx}'), f"File not found: {image_path}"))
                continue
            
            # Check if it's a valid image
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Verify it's a valid image
                
                # Check file size (optional - warn if too large)
                file_size = os.path.getsize(image_path)
                if file_size > 50 * 1024 * 1024:  # 50MB
                    print(f"⚠️  WARNING: Large image file: {image_path} ({file_size/1024/1024:.1f}MB)")
                
                valid_count += 1
                
            except Exception as img_error:
                invalid_count += 1
                invalid_samples.append((idx, item.get('id', f'sample_{idx}'), f"Invalid image: {str(img_error)}"))
                continue
                
        except Exception as e:
            invalid_count += 1
            invalid_samples.append((idx, item.get('id', f'sample_{idx}'), f"Processing error: {str(e)}"))
            continue
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(data)} samples verified...")
    
    # Print results
    print(f"\n📈 VERIFICATION RESULTS:")
    print(f"  ✅ Valid images: {valid_count}")
    print(f"  ❌ Invalid images: {invalid_count}")
    print(f"  📊 Success rate: {(valid_count/(valid_count+invalid_count)*100):.1f}%" if (valid_count+invalid_count) > 0 else "  📊 Success rate: 0%")
    
    # Show invalid samples
    if invalid_samples:
        print(f"\n❌ INVALID SAMPLES (showing first 10):")
        for idx, sample_id, error in invalid_samples[:10]:
            print(f"  Row {idx} (ID: {sample_id}): {error}")
        
        if len(invalid_samples) > 10:
            print(f"  ... and {len(invalid_samples) - 10} more invalid samples")
    
    return valid_count, invalid_count, invalid_samples

def main():
    """Main function to verify both train.json and val.json"""
    
    print("🔍 STAGE1 DATA IMAGE PATH VERIFICATION")
    print("="*60)
    
    # Define paths
    base_dir = Path("stage1_data")
    train_file = base_dir / "train.json"
    val_file = base_dir / "val.json"
    
    # Check if stage1_data directory exists
    if not base_dir.exists():
        print(f"❌ ERROR: Directory not found: {base_dir}")
        print("Please ensure you're running this script from the correct directory.")
        sys.exit(1)
    
    total_valid = 0
    total_invalid = 0
    all_invalid_samples = []
    
    # Verify train.json
    train_valid, train_invalid, train_invalid_samples = verify_image_paths(train_file)
    total_valid += train_valid
    total_invalid += train_invalid
    all_invalid_samples.extend([("train", sample) for sample in train_invalid_samples])
    
    # Verify val.json
    val_valid, val_invalid, val_invalid_samples = verify_image_paths(val_file)
    total_valid += val_valid
    total_invalid += val_invalid
    all_invalid_samples.extend([("val", sample) for sample in val_invalid_samples])
    
    # Overall summary
    print(f"\n{'='*60}")
    print(f"📊 OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  ✅ Total valid images: {total_valid}")
    print(f"  ❌ Total invalid images: {total_invalid}")
    print(f"  📊 Overall success rate: {(total_valid/(total_valid+total_invalid)*100):.1f}%" if (total_valid+total_invalid) > 0 else "  📊 Overall success rate: 0%")
    
    if total_invalid > 0:
        print(f"\n⚠️  WARNING: {total_invalid} invalid image paths found!")
        print("These need to be fixed before training can proceed successfully.")
        
        # Save invalid samples to file
        invalid_file = "invalid_image_paths.log"
        with open(invalid_file, 'w') as f:
            f.write("INVALID IMAGE PATHS REPORT\n")
            f.write("="*50 + "\n\n")
            for file_type, (idx, sample_id, error) in all_invalid_samples:
                f.write(f"{file_type.upper()}: Row {idx} (ID: {sample_id}): {error}\n")
        
        print(f"📝 Detailed report saved to: {invalid_file}")
    else:
        print(f"\n🎉 SUCCESS: All image paths are valid!")
    
    return total_invalid == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
