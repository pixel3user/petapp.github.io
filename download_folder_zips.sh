#!/bin/bash

# Simple script to download all zip files from the specific Google Drive folder
# Folder ID: 1Ih8M4V5P7BrDZc_ggY_ALmeib_cB6X4V

echo "🚀 Downloading zip files from Google Drive folder..."
echo "Folder ID: 1Ih8M4V5P7BrDZc_ggY_ALmeib_cB6X4V"
echo ""

# Run the Python script with the specific folder ID
python3 download_from_gdrive.py --folder-id 1Ih8M4V5P7BrDZc_ggY_ALmeib_cB6X4V

echo ""
echo "✅ Download complete! Check the root directory for extracted folders."







