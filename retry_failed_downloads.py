# Failed Download Handler Script
import pandas as pd
import requests
import os
from pathlib import Path
import time

def retry_failed_downloads(csv_path="stage1_data/unified_dataset.csv", max_retries=3):
    """Retry failed downloads with different strategies"""
    
    print("="*60)
    print("RETRYING FAILED DOWNLOADS")
    print("="*60)
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['image_path'])
    
    # Find URL samples that might have failed
    url_samples = df[df['image_path'].str.startswith('http')]
    
    # Check which URLs don't have corresponding downloaded files
    failed_urls = []
    downloaded_dir = "downloaded_images"
    
    for idx, row in url_samples.iterrows():
        url = row['image_path']
        dataset = row.get('dataset', 'unknown')
        disease = row.get('disease', 'unknown')
        
        # Create expected filename
        safe_disease = "".join(c for c in str(disease) if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{dataset}_{safe_disease}_{idx}.jpg"
        local_path = os.path.join(downloaded_dir, filename)
        
        # If file doesn't exist, it's a failed download
        if not os.path.exists(local_path):
            failed_urls.append((idx, url, dataset, disease, local_path))
    
    print(f"Found {len(failed_urls)} failed downloads to retry")
    
    if len(failed_urls) == 0:
        print("No failed downloads found!")
        return
    
    # Retry failed downloads
    success_count = 0
    still_failed = []
    
    for idx, url, dataset, disease, local_path in failed_urls:
        print(f"Retrying: {url}")
        
        success = False
        for attempt in range(max_retries):
            try:
                # Try different strategies
                if attempt == 0:
                    # Standard request
                    response = requests.get(url, timeout=30)
                elif attempt == 1:
                    # With headers
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(url, headers=headers, timeout=30)
                else:
                    # Longer timeout
                    response = requests.get(url, timeout=60)
                
                response.raise_for_status()
                
                # Save image
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"  ✓ Success on attempt {attempt + 1}")
                success_count += 1
                success = True
                break
                
            except Exception as e:
                print(f"  ✗ Attempt {attempt + 1} failed: {str(e)[:100]}...")
                time.sleep(2)  # Wait before retry
        
        if not success:
            still_failed.append((idx, url, dataset, disease))
    
    print(f"\nRetry Results:")
    print(f"Successfully downloaded: {success_count}")
    print(f"Still failed: {len(still_failed)}")
    
    if still_failed:
        print(f"\nStill failed URLs:")
        for idx, url, dataset, disease in still_failed[:10]:  # Show first 10
            print(f"  {url}")
        
        # Save failed URLs to file for manual inspection
        with open('failed_downloads.txt', 'w') as f:
            f.write("Failed Downloads Report\n")
            f.write("=" * 50 + "\n\n")
            for idx, url, dataset, disease in still_failed:
                f.write(f"Index: {idx}\n")
                f.write(f"URL: {url}\n")
                f.write(f"Dataset: {dataset}\n")
                f.write(f"Disease: {disease}\n")
                f.write("-" * 30 + "\n")
        
        print(f"\nFailed URLs saved to: failed_downloads.txt")
    
    return success_count, len(still_failed)

def remove_failed_samples_from_dataset():
    """Remove samples with failed downloads from the Qwen2-VL datasets"""
    
    print("\n" + "="*60)
    print("CLEANING UP DATASET")
    print("="*60)
    
    # Load the datasets
    with open('qwen2vl_train.json', 'r') as f:
        train_data = json.load(f)
    
    with open('qwen2vl_val.json', 'r') as f:
        val_data = json.load(f)
    
    print(f"Original train samples: {len(train_data)}")
    print(f"Original val samples: {len(val_data)}")
    
    # Filter out samples with missing images
    clean_train = []
    clean_val = []
    
    for item in train_data:
        if os.path.exists(item['image']):
            clean_train.append(item)
    
    for item in val_data:
        if os.path.exists(item['image']):
            clean_val.append(item)
    
    print(f"Clean train samples: {len(clean_train)}")
    print(f"Clean val samples: {len(clean_val)}")
    
    # Save cleaned datasets
    with open('qwen2vl_train_clean.json', 'w') as f:
        json.dump(clean_train, f, indent=2)
    
    with open('qwen2vl_val_clean.json', 'w') as f:
        json.dump(clean_val, f, indent=2)
    
    print(f"\nCleaned datasets saved:")
    print(f"- qwen2vl_train_clean.json ({len(clean_train)} samples)")
    print(f"- qwen2vl_val_clean.json ({len(clean_val)} samples)")
    
    return clean_train, clean_val

# Run the retry process
if __name__ == "__main__":
    import json
    
    # Retry failed downloads
    success_count, failed_count = retry_failed_downloads()
    
    # Clean up datasets
    clean_train, clean_val = remove_failed_samples_from_dataset()
    
    print(f"\n" + "="*60)
    print("CLEANUP COMPLETE!")
    print("="*60)
    print(f"Use the clean datasets for training:")
    print(f"- qwen2vl_train_clean.json")
    print(f"- qwen2vl_val_clean.json")
