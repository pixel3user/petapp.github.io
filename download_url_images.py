# URL Image Download Script
import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
import time
from pathlib import Path

def download_url_images(csv_path, output_dir="downloaded_images"):
    """Download all URL images and update the CSV with local paths"""
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Filter URL samples (handle NaN values)
    df = df.dropna(subset=['image_path'])  # Remove rows with NaN image_path
    url_samples = df[df['image_path'].str.startswith('http')]
    print(f"Found {len(url_samples)} URL images to download")
    
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
    
    # Save updated dataset
    output_csv = csv_path.replace('.csv', '_with_downloaded_images.csv')
    df.to_csv(output_csv, index=False)
    
    print(f"\nDownload complete!")
    print(f"Downloaded: {downloaded_count}")
    print(f"Failed: {failed_count}")
    print(f"Updated dataset saved to: {output_csv}")
    
    return df

# Usage example:
if __name__ == "__main__":
    download_url_images('stage1_data/unified_dataset.csv')
