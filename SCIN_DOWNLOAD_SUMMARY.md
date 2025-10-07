# SCIN Dataset Download Summary

## Dataset Information
- **Name**: SCIN (Scientific Case Investigation Network)
- **Source**: Google Cloud Storage - `gs://dx-scin-public-data/`
- **GitHub Repository**: https://github.com/google-research-datasets/scin
- **Download Date**: September 15, 2025
- **Location**: `/teamspace/studios/this_studio/data/scin/`
- **Last Updated**: March 2024

## Dataset Structure
The dataset contains scientific case investigation data with the following components:

### 1. Images
**Location**: `data/scin/dataset/images/`
- **Count**: 10,379 images
- **Format**: PNG files
- **Size**: ~11.7 GB
- **Description**: Scientific case images for investigation

### 2. Data Files
**Location**: `data/scin/dataset/`

#### CSV Files:
- **`scin_cases.csv`** (1.3 MB, 5,034 rows)
  - Main cases data
  - Contains case information and metadata

- **`scin_labels.csv`** (896 KB, 5,034 rows)
  - Labels for the cases
  - Corresponds to the cases in scin_cases.csv

- **`scin_app_questions.csv`** (2.3 KB, 99 rows)
  - Application questions
  - Questions related to case applications

- **`scin_label_questions.csv`** (3.3 KB, 43 rows)
  - Label questions
  - Questions related to labeling tasks

### 3. Documentation
- **`README.md`** (123 bytes)
  - Basic information and GitHub repository link

## Total Download Statistics
- **Total Files**: 10,384 files
- **Total Size**: 12 GB
- **Main Components**:
  - Images: ~11.7 GB (10,379 PNG files)
  - CSV data files: ~2.2 MB (4 files)
  - Documentation: 123 bytes

## Dataset Purpose
The SCIN dataset appears to be designed for scientific case investigation tasks, likely involving:
- Visual analysis of scientific cases
- Question-answering about scientific scenarios
- Case labeling and classification
- Application of scientific knowledge

## Usage Notes
- The dataset is publicly available from Google Research
- Images are in PNG format and appear to be scientific case visualizations
- The CSV files contain structured data about cases, labels, and questions
- For more detailed information, refer to the GitHub repository: https://github.com/google-research-datasets/scin

## Citation
When using this dataset, please cite the original work and reference the GitHub repository:
- GitHub Repository: https://github.com/google-research-datasets/scin
- Last Updated: March 2024
