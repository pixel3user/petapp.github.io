# Dataset Verification and Download Guide

This guide explains how to use the `check_and_download_datasets.py` script to verify all datasets are properly installed and download missing ones.

## Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_kagglehub.txt
   ```

## Usage Options

### 1. Check All Datasets (Basic Check)
```bash
python check_and_download_datasets.py
```
This will:
- Check if `data/` and `downloaded_images/` directories exist
- Verify all expected datasets are present
- Generate a detailed report
- Create missing directories if needed

### 2. Check and Download Missing Datasets
```bash
python check_and_download_datasets.py --download-missing
```
This will:
- Perform all checks from option 1
- Automatically download DermNet dataset if missing
- Re-verify after download

### 3. Save Detailed Results
```bash
python check_and_download_datasets.py --save-results
```
This will:
- Perform all checks
- Save detailed results to `dataset_verification_results.json`

### 4. Verbose Logging
```bash
python check_and_download_datasets.py --verbose
```
This will:
- Show detailed logging information
- Help with debugging issues

### 5. All Options Combined
```bash
python check_and_download_datasets.py --download-missing --save-results --verbose
```

## Expected Datasets

The script checks for these datasets:

1. **DDI (Diverse Dermatology Images)**
   - Location: `data/ddidiversedermatologyimages/`
   - Files: Image files (jpg, jpeg, png)

2. **DermaVQA**
   - Location: `data/dermavqa/`
   - Files: Dataset structure with images and metadata

3. **Fitzpatrick17k**
   - Location: `data/fitzpatrick17k/`
   - Files: `fitzpatrick17k.csv`

4. **SCIN (Skin Cancer Image Network)**
   - Location: `data/scin/`
   - Files: Dataset with images and labels

5. **SkinCap**
   - Location: `data/skincap/`
   - Files: `skincap_v240623.csv`

6. **DermNet** ⭐
   - Location: `.cache/kagglehub/datasets/shubhamgoel27/dermnet/`
   - Download: Automatic via kagglehub
   - Dataset: `shubhamgoel27/dermnet`

## Output

The script provides:

### Console Output
- Real-time status updates
- Detailed verification report
- Recommendations for missing datasets
- Summary statistics

### Log File
- Detailed log saved to `dataset_check.log`
- Includes all operations and errors

### JSON Results (Optional)
- Detailed results saved to `dataset_verification_results.json`
- Machine-readable format for further processing

## Example Output

```
================================================================================
DATASET VERIFICATION REPORT
================================================================================

SUMMARY:
  Total datasets: 6
  Complete: 4
  Partial: 1
  Missing: 1
  Errors: 0

DETAILED RESULTS:
--------------------------------------------------------------------------------
✅ DDI (Diverse Dermatology Images) (ddi)
    Status: complete
    Total files: 656

✅ DermaVQA (dermavqa)
    Status: complete
    Total files: 1234

⚠️ Fitzpatrick17k (fitzpatrick)
    Status: partial
    Total files: 15000
    Missing files: 1
      - fitzpatrick17k.csv

❌ DermNet (dermnet)
    Status: missing
    Missing files: 1
      - .cache/kagglehub/datasets/shubhamgoel27/dermnet/

RECOMMENDATIONS:
--------------------------------------------------------------------------------
Missing datasets detected:
  - DermNet: Run with --download-missing to download via kagglehub

Partially installed datasets:
  - Fitzpatrick17k: Some files missing, check installation
```

## Troubleshooting

### Common Issues

1. **kagglehub not installed**
   ```bash
   pip install kagglehub
   ```

2. **Permission errors**
   - Ensure you have write permissions to the project directory
   - Check if directories can be created

3. **Network issues during download**
   - Check internet connection
   - Kaggle API credentials may be required for some datasets

4. **Missing directories**
   - The script will automatically create `data/` and `downloaded_images/` if missing

### Exit Codes
- `0`: All datasets are properly installed
- `1`: Some datasets are missing or have errors

## Integration

This script can be integrated into your workflow:

```bash
# Check before training
python check_and_download_datasets.py --download-missing

# If successful, proceed with training
if [ $? -eq 0 ]; then
    echo "All datasets ready, starting training..."
    python stage1_data_preparation.py
else
    echo "Dataset issues detected, please fix before training"
    exit 1
fi
```

## Notes

- The script is safe to run multiple times
- It won't re-download existing datasets
- DermNet download may take time depending on internet speed
- All operations are logged for debugging







