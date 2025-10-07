#!/usr/bin/env python3
"""
Script to download the DermaVQA dataset from OSF (Open Science Framework)
Dataset: Visual Question Answering in Dermatology (dermavqa)
URL: https://osf.io/72rp3/
"""

import os
import sys
import requests
import json
from urllib.parse import urljoin

def download_file(url, filepath):
    """Download a file from URL to filepath"""
    try:
        print(f"Downloading {os.path.basename(filepath)}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(filepath)
        print(f"✓ Downloaded {os.path.basename(filepath)} ({file_size:,} bytes)")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {os.path.basename(filepath)}: {e}")
        return False

def download_dermavqa_dataset():
    """Download the DermaVQA dataset from OSF to the data folder"""
    
    # Set the output directory
    output_dir = "/teamspace/studios/this_studio/data/dermavqa"
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading DermaVQA dataset from OSF...")
    print("Dataset: Visual Question Answering in Dermatology")
    print("URL: https://osf.io/72rp3/")
    print(f"Output directory: {output_dir}")
    
    # Files to download with their direct download URLs
    files_to_download = [
        {
            "name": "dermavqa_metadata.json",
            "url": "https://osf.io/download/5y6pt/",
            "path": os.path.join(output_dir, "dermavqa_metadata.json")
        },
        {
            "name": "README.md",
            "url": "https://osf.io/download/25ve8/",
            "path": os.path.join(output_dir, "README.md")
        },
        {
            "name": "LICENSE",
            "url": "https://osf.io/download/67n5p/",
            "path": os.path.join(output_dir, "LICENSE")
        },
        {
            "name": "CODE_OF_CONDUCT.md",
            "url": "https://osf.io/download/rm2g6/",
            "path": os.path.join(output_dir, "CODE_OF_CONDUCT.md")
        },
        {
            "name": "SECURITY.md",
            "url": "https://osf.io/download/n67qx/",
            "path": os.path.join(output_dir, "SECURITY.md")
        },
        {
            "name": "dermavqa_example.png",
            "url": "https://osf.io/download/hqzdt/",
            "path": os.path.join(output_dir, "dermavqa_example.png")
        },
        {
            "name": "iiyi_multipartyconversation-to-queryresponsepair_guidelines_release.pdf",
            "url": "https://osf.io/download/m8vsa/",
            "path": os.path.join(output_dir, "iiyi_multipartyconversation-to-queryresponsepair_guidelines_release.pdf")
        }
    ]
    
    downloaded_count = 0
    total_files = len(files_to_download)
    
    # Download main files
    for file_info in files_to_download:
        if download_file(file_info["url"], file_info["path"]):
            downloaded_count += 1
    
    print(f"\nDownloaded {downloaded_count}/{total_files} main files")
    
    # Now let's try to get the data folders
    print("\nAttempting to download data folders...")
    
    try:
        # Get files from iiyi folder
        iiyi_url = "https://api.osf.io/v2/nodes/72rp3/files/osfstorage/6682ed6e1960ff002458eb52/"
        response = requests.get(iiyi_url)
        response.raise_for_status()
        
        iiyi_data = response.json()
        iiyi_files = iiyi_data['data']
        
        print(f"Found {len(iiyi_files)} files in iiyi folder")
        
        for file_info in iiyi_files:
            if file_info['attributes']['kind'] == 'file':
                file_name = file_info['attributes']['name']
                file_url = file_info['links']['download']
                file_path = os.path.join(output_dir, "data", "iiyi", file_name)
                
                if download_file(file_url, file_path):
                    downloaded_count += 1
        
        # Get files from reddit folder
        reddit_url = "https://api.osf.io/v2/nodes/72rp3/files/osfstorage/6682ed4fede75d0029693e72/"
        response = requests.get(reddit_url)
        response.raise_for_status()
        
        reddit_data = response.json()
        reddit_files = reddit_data['data']
        
        print(f"Found {len(reddit_files)} files in reddit folder")
        
        for file_info in reddit_files:
            if file_info['attributes']['kind'] == 'file':
                file_name = file_info['attributes']['name']
                file_url = file_info['links']['download']
                file_path = os.path.join(output_dir, "data", "reddit", file_name)
                
                if download_file(file_url, file_path):
                    downloaded_count += 1
        
    except Exception as e:
        print(f"Error downloading data folders: {e}")
    
    print(f"\nTotal downloaded: {downloaded_count} files")
    print(f"Dataset saved to: {output_dir}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, output_dir)
            file_size = os.path.getsize(file_path)
            print(f"  - {relative_path} ({file_size:,} bytes)")
    
    return downloaded_count > 0

if __name__ == "__main__":
    success = download_dermavqa_dataset()
    if success:
        print("\n✓ DermaVQA dataset download completed successfully!")
    else:
        print("\n✗ DermaVQA dataset download failed!")
        sys.exit(1)