# DermaVQA Dataset Download Summary

## Dataset Information
- **Name**: Visual Question Answering in Dermatology (DermaVQA)
- **Source**: OSF (Open Science Framework) - https://osf.io/72rp3/
- **Download Date**: September 15, 2025
- **Location**: `/teamspace/studios/this_studio/data/dermavqa/`

## Dataset Structure
The dataset contains two main data sources:

### 1. IIYI (Itchy Itchy Yucky Yucky) Data
**Location**: `data/dermavqa/data/iiyi/`

**Files**:
- `images_final.zip` (1.1 GB) - Extracted to `images_final/` folder
- `df_userinfo.csv` (400 KB) - User information
- `df_users_map.csv` (3 MB) - User mapping data
- `valid_ht.json` (278 KB) - Validation set
- `valid_ht_v2.json` (278 KB) - Validation set v2
- `test_ht.json` (525 KB) - Test set
- `test_ht_spanishtestsetcorrected.json` (527 KB) - Spanish test set
- `instanceid2encounterid.json` (25 KB) - Instance to encounter mapping
- `df_mediqa-m3g-final.csv` (40 KB) - MEDIQA-M3G data
- `split2encounterids.json` (17 KB) - Split to encounter mapping

### 2. Reddit Data
**Location**: `data/dermavqa/data/reddit/`

**Files**:
- `train_answersonly.json` (282 KB) - Training set
- `valid_answersonly.json` (40 KB) - Validation set
- `test_answersonly.json` (141 KB) - Test set
- `df_mediqa-magic-final.csv` (21 KB) - MEDIQA-MAGIC data
- `download_data.py` (2 KB) - Data download script

### 3. Documentation and Metadata
**Location**: `data/dermavqa/`

**Files**:
- `README.md` (8 KB) - Dataset documentation
- `dermavqa_metadata.json` (13 KB) - Dataset metadata
- `LICENSE` (19 KB) - License information
- `CODE_OF_CONDUCT.md` (444 bytes) - Code of conduct
- `SECURITY.md` (3 KB) - Security policy
- `dermavqa_example.png` (397 KB) - Example image
- `iiyi_multipartyconversation-to-queryresponsepair_guidelines_release.pdf` (890 KB) - Guidelines

## Total Download Statistics
- **Total Files**: 22 files
- **Total Size**: ~1.1 GB (including extracted images)
- **Main Components**: 
  - Images: ~1.1 GB (extracted from zip)
  - JSON data files: ~2.5 MB
  - CSV data files: ~3.5 MB
  - Documentation: ~1.3 MB

## Usage Notes
- The dataset is publicly available under CC-By Attribution 4.0 International license
- Images are organized in `images_final/` folder with subdirectories for different splits
- JSON files contain question-answer pairs and metadata
- CSV files contain user and encounter information
- The dataset supports visual question answering tasks in dermatology

## Citation
When using this dataset, please cite the original paper and reference the OSF project:
- OSF Project: https://osf.io/72rp3/
- License: CC-By Attribution 4.0 International
