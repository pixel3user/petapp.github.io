#!/usr/bin/env python3
"""
Dataset Verification and Download Script
=======================================

This script checks for the existence of all required datasets and downloads
missing ones, particularly the DermNet dataset from Kaggle.

Features:
- Checks for data/ and downloaded_images/ folders
- Verifies all required datasets are present
- Downloads DermNet dataset using kagglehub if missing
- Provides detailed status report
- Handles errors gracefully

Usage:
    python check_and_download_datasets.py [--download-missing] [--verbose]

Requirements:
    pip install kagglehub
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_check.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
ROOT_DIR = Path('/teamspace/studios/this_studio')
DATA_DIR = ROOT_DIR / 'data'
DOWNLOADED_IMAGES_DIR = ROOT_DIR / 'downloaded_images'

# Expected datasets and their locations
EXPECTED_DATASETS = {
    'ddi': {
        'name': 'DDI (Diverse Dermatology Images)',
        'data_path': DATA_DIR / 'ddidiversedermatologyimages',
        'downloaded_path': DOWNLOADED_IMAGES_DIR,
        'required_files': ['*.jpg', '*.jpeg', '*.png'],
        'description': 'Diverse Dermatology Images dataset'
    },
    'dermavqa': {
        'name': 'DermaVQA',
        'data_path': DATA_DIR / 'dermavqa',
        'downloaded_path': DOWNLOADED_IMAGES_DIR,
        'required_files': ['data/iiyi/images_final/'],
        'description': 'Dermatology Visual Question Answering dataset'
    },
    'fitzpatrick': {
        'name': 'Fitzpatrick17k',
        'data_path': DATA_DIR / 'fitzpatrick17k',
        'downloaded_path': DOWNLOADED_IMAGES_DIR,
        'required_files': ['fitzpatrick17k.csv'],
        'description': 'Fitzpatrick skin type classification dataset'
    },
    'scin': {
        'name': 'SCIN',
        'data_path': DATA_DIR / 'scin',
        'downloaded_path': DOWNLOADED_IMAGES_DIR,
        'required_files': ['dataset/images/'],
        'description': 'Skin Cancer Image Network dataset'
    },
    'skincap': {
        'name': 'SkinCap',
        'data_path': DATA_DIR / 'skincap',
        'downloaded_path': DOWNLOADED_IMAGES_DIR,
        'required_files': ['skincap_v240623.csv'],
        'description': 'Skin Caption dataset'
    },
    'dermnet': {
        'name': 'DermNet',
        'data_path': None,  # Uses kagglehub cache
        'downloaded_path': None,  # Uses kagglehub cache
        'required_files': ['.cache/kagglehub/datasets/shubhamgoel27/dermnet/'],
        'description': 'DermNet dataset from Kaggle (uses kagglehub)',
        'kaggle_dataset': 'shubhamgoel27/dermnet'
    }
}


class DatasetChecker:
    """Dataset verification and download manager"""
    
    def __init__(self, root_dir: Path = ROOT_DIR):
        self.root_dir = root_dir
        self.data_dir = root_dir / 'data'
        self.downloaded_images_dir = root_dir / 'downloaded_images'
        self.results = {}
        
    def check_directory_structure(self) -> Dict[str, bool]:
        """Check if required directories exist"""
        logger.info("Checking directory structure...")
        
        directories = {
            'root': self.root_dir.exists(),
            'data': self.data_dir.exists(),
            'downloaded_images': self.downloaded_images_dir.exists()
        }
        
        for name, exists in directories.items():
            status = "✓" if exists else "❌"
            logger.info(f"  {status} {name}/ directory: {exists}")
            
            if not exists and name in ['data', 'downloaded_images']:
                logger.warning(f"  Missing {name}/ directory - creating it...")
                try:
                    if name == 'data':
                        self.data_dir.mkdir(exist_ok=True)
                    elif name == 'downloaded_images':
                        self.downloaded_images_dir.mkdir(exist_ok=True)
                    logger.info(f"  ✓ Created {name}/ directory")
                    directories[name] = True
                except Exception as e:
                    logger.error(f"  ❌ Failed to create {name}/ directory: {e}")
        
        return directories
    
    def check_dataset_files(self, dataset_key: str, dataset_info: Dict) -> Dict[str, any]:
        """Check if a specific dataset is properly installed"""
        logger.info(f"Checking {dataset_info['name']}...")
        
        result = {
            'name': dataset_info['name'],
            'key': dataset_key,
            'data_exists': False,
            'downloaded_exists': False,
            'files_found': [],
            'files_missing': [],
            'total_files': 0,
            'status': 'missing'
        }
        
        # Check data directory
        if dataset_info['data_path'] and dataset_info['data_path'].exists():
            result['data_exists'] = True
            logger.info(f"  ✓ Data directory exists: {dataset_info['data_path']}")
        else:
            logger.warning(f"  ❌ Data directory missing: {dataset_info['data_path']}")
        
        # Check downloaded images directory
        if dataset_info['downloaded_path'] and dataset_info['downloaded_path'].exists():
            result['downloaded_exists'] = True
            logger.info(f"  ✓ Downloaded images directory exists: {dataset_info['downloaded_path']}")
        else:
            logger.warning(f"  ❌ Downloaded images directory missing: {dataset_info['downloaded_path']}")
        
        # Check required files
        for required_file in dataset_info['required_files']:
            if self._check_file_or_pattern(required_file, dataset_info):
                result['files_found'].append(required_file)
                logger.info(f"  ✓ Found: {required_file}")
            else:
                result['files_missing'].append(required_file)
                logger.warning(f"  ❌ Missing: {required_file}")
        
        # Determine overall status
        if result['files_found'] and not result['files_missing']:
            result['status'] = 'complete'
        elif result['files_found']:
            result['status'] = 'partial'
        else:
            result['status'] = 'missing'
        
        # Count total files for datasets with images
        if dataset_key != 'dermnet':  # Skip counting for kagglehub datasets
            result['total_files'] = self._count_files_in_dataset(dataset_info)
        
        return result
    
    def _check_file_or_pattern(self, pattern: str, dataset_info: Dict) -> bool:
        """Check if a file or pattern exists"""
        # Handle kagglehub cache paths
        if pattern.startswith('.cache/kagglehub/'):
            cache_path = self.root_dir / pattern
            return cache_path.exists()
        
        # Handle data directory paths
        if dataset_info['data_path']:
            full_path = dataset_info['data_path'] / pattern
            if full_path.exists():
                return True
        
        # Handle downloaded images directory
        if dataset_info['downloaded_path']:
            # Check for dataset-specific files in downloaded_images
            if '*' in pattern:  # Pattern matching
                import glob
                search_path = dataset_info['downloaded_path'] / pattern
                matches = list(glob.glob(str(search_path)))
                return len(matches) > 0
            else:
                full_path = dataset_info['downloaded_path'] / pattern
                if full_path.exists():
                    return True
        
        return False
    
    def _count_files_in_dataset(self, dataset_info: Dict) -> int:
        """Count total files in a dataset"""
        count = 0
        
        # Count files in data directory
        if dataset_info['data_path'] and dataset_info['data_path'].exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.jpeg']:
                import glob
                pattern = str(dataset_info['data_path'] / '**' / ext)
                count += len(glob.glob(pattern, recursive=True))
        
        # Count files in downloaded_images directory
        if dataset_info['downloaded_path'] and dataset_info['downloaded_path'].exists():
            # Look for dataset-specific files
            dataset_name = dataset_info['name'].lower().replace(' ', '_')
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.jpeg']:
                import glob
                pattern = str(dataset_info['downloaded_path'] / f'*{dataset_name}*{ext}')
                count += len(glob.glob(pattern))
        
        return count
    
    def download_dermnet(self) -> bool:
        """Download DermNet dataset using kagglehub"""
        logger.info("Downloading DermNet dataset using kagglehub...")
        
        try:
            import kagglehub
            
            # Download the dataset
            logger.info("Starting DermNet download...")
            path = kagglehub.dataset_download("shubhamgoel27/dermnet")
            
            logger.info(f"✓ DermNet downloaded successfully!")
            logger.info(f"  Path to dataset files: {path}")
            
            # Verify the download
            if Path(path).exists():
                logger.info(f"✓ Dataset verification successful")
                return True
            else:
                logger.error(f"❌ Dataset verification failed - path not found: {path}")
                return False
                
        except ImportError:
            logger.error("❌ kagglehub not installed. Please install it with: pip install kagglehub")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to download DermNet dataset: {e}")
            return False
    
    def check_stage1_test_data(self) -> Dict[str, any]:
        """Check stage1_test_data folder for dataset references"""
        logger.info("Checking stage1_test_data folder...")
        
        test_data_dir = self.root_dir / 'stage1_test_data'
        if not test_data_dir.exists():
            logger.warning("stage1_test_data directory not found")
            return {'status': 'missing', 'error': 'Directory not found'}
        
        results = {
            'status': 'found',
            'files': {},
            'dataset_counts': {},
            'path_counts': {}
        }
        
        # Check train.json and val.json
        for filename in ['train.json', 'val.json']:
            file_path = test_data_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Count datasets
                    dataset_counts = {}
                    path_counts = {}
                    
                    for item in data:
                        # Count datasets
                        if 'metadata' in item and 'dataset' in item['metadata']:
                            dataset = item['metadata']['dataset']
                            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
                        
                        # Count path types
                        if 'image' in item:
                            path = item['image']
                            if path.startswith('.cache/'):
                                path_type = '.cache'
                            elif path.startswith('data/'):
                                path_type = 'data'
                            elif path.startswith('downloaded_images/'):
                                path_type = 'downloaded_images'
                            else:
                                path_type = 'other'
                            path_counts[path_type] = path_counts.get(path_type, 0) + 1
                    
                    results['files'][filename] = {
                        'total_samples': len(data),
                        'dataset_counts': dataset_counts,
                        'path_counts': path_counts
                    }
                    
                    logger.info(f"  ✓ {filename}: {len(data)} samples")
                    for dataset, count in dataset_counts.items():
                        logger.info(f"    - {dataset}: {count} samples")
                    for path_type, count in path_counts.items():
                        logger.info(f"    - {path_type}: {count} samples")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error reading {filename}: {e}")
                    results['files'][filename] = {'error': str(e)}
            else:
                logger.warning(f"  ❌ {filename} not found")
                results['files'][filename] = {'error': 'File not found'}
        
        return results
    
    def check_all_datasets(self) -> Dict[str, Dict]:
        """Check all expected datasets"""
        logger.info("Checking all datasets...")
        
        results = {}
        
        for dataset_key, dataset_info in EXPECTED_DATASETS.items():
            try:
                result = self.check_dataset_files(dataset_key, dataset_info)
                results[dataset_key] = result
            except Exception as e:
                logger.error(f"❌ Error checking {dataset_info['name']}: {e}")
                results[dataset_key] = {
                    'name': dataset_info['name'],
                    'key': dataset_key,
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def generate_report(self, results: Dict[str, Dict], test_data_results: Dict = None) -> str:
        """Generate a comprehensive report"""
        report = []
        report.append("=" * 80)
        report.append("DATASET VERIFICATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_datasets = len(results)
        complete_datasets = sum(1 for r in results.values() if r.get('status') == 'complete')
        partial_datasets = sum(1 for r in results.values() if r.get('status') == 'partial')
        missing_datasets = sum(1 for r in results.values() if r.get('status') == 'missing')
        error_datasets = sum(1 for r in results.values() if r.get('status') == 'error')
        
        report.append("SUMMARY:")
        report.append(f"  Total datasets: {total_datasets}")
        report.append(f"  Complete: {complete_datasets}")
        report.append(f"  Partial: {partial_datasets}")
        report.append(f"  Missing: {missing_datasets}")
        report.append(f"  Errors: {error_datasets}")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 80)
        
        for dataset_key, result in results.items():
            status_icon = {
                'complete': '✅',
                'partial': '⚠️',
                'missing': '❌',
                'error': '💥'
            }.get(result.get('status', 'unknown'), '❓')
            
            report.append(f"{status_icon} {result['name']} ({dataset_key})")
            report.append(f"    Status: {result.get('status', 'unknown')}")
            
            if 'total_files' in result and result['total_files'] > 0:
                report.append(f"    Total files: {result['total_files']}")
            
            if result.get('files_found'):
                report.append(f"    Found files: {len(result['files_found'])}")
                for file in result['files_found'][:3]:  # Show first 3
                    report.append(f"      - {file}")
                if len(result['files_found']) > 3:
                    report.append(f"      ... and {len(result['files_found']) - 3} more")
            
            if result.get('files_missing'):
                report.append(f"    Missing files: {len(result['files_missing'])}")
                for file in result['files_missing']:
                    report.append(f"      - {file}")
            
            if 'error' in result:
                report.append(f"    Error: {result['error']}")
            
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 80)
        
        if missing_datasets > 0:
            report.append("Missing datasets detected:")
            for dataset_key, result in results.items():
                if result.get('status') == 'missing':
                    if dataset_key == 'dermnet':
                        report.append(f"  - {result['name']}: Run with --download-missing to download via kagglehub")
                    else:
                        report.append(f"  - {result['name']}: Check download scripts or manual download required")
            report.append("")
        
        if partial_datasets > 0:
            report.append("Partially installed datasets:")
            for dataset_key, result in results.items():
                if result.get('status') == 'partial':
                    report.append(f"  - {result['name']}: Some files missing, check installation")
            report.append("")
        
        # Add stage1_test_data information
        if test_data_results and test_data_results.get('status') == 'found':
            report.append("STAGE1_TEST_DATA ANALYSIS:")
            report.append("-" * 80)
            
            for filename, file_info in test_data_results.get('files', {}).items():
                if 'error' not in file_info:
                    report.append(f"📁 {filename}:")
                    report.append(f"    Total samples: {file_info['total_samples']}")
                    
                    if 'dataset_counts' in file_info:
                        report.append("    Dataset breakdown:")
                        for dataset, count in file_info['dataset_counts'].items():
                            report.append(f"      - {dataset}: {count} samples")
                    
                    if 'path_counts' in file_info:
                        report.append("    Path breakdown:")
                        for path_type, count in file_info['path_counts'].items():
                            report.append(f"      - {path_type}: {count} samples")
                    report.append("")
                else:
                    report.append(f"❌ {filename}: {file_info['error']}")
                    report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Dict], filename: str = "dataset_verification_results.json"):
        """Save results to JSON file"""
        output_file = self.root_dir / filename
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"✓ Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Check and download datasets")
    parser.add_argument('--download-missing', action='store_true', 
                       help='Download missing datasets (currently supports DermNet)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--save-results', action='store_true',
                       help='Save detailed results to JSON file')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting dataset verification...")
    
    # Initialize checker
    checker = DatasetChecker()
    
    # Check directory structure
    dir_results = checker.check_directory_structure()
    
    # Check all datasets
    results = checker.check_all_datasets()
    
    # Check stage1_test_data folder
    test_data_results = checker.check_stage1_test_data()
    
    # Download missing datasets if requested
    if args.download_missing:
        logger.info("Checking for missing datasets to download...")
        
        for dataset_key, result in results.items():
            if result.get('status') == 'missing' and dataset_key == 'dermnet':
                logger.info(f"Downloading {result['name']}...")
                success = checker.download_dermnet()
                if success:
                    # Re-check the dataset
                    dataset_info = EXPECTED_DATASETS[dataset_key]
                    results[dataset_key] = checker.check_dataset_files(dataset_key, dataset_info)
                    logger.info(f"✓ {result['name']} download completed")
                else:
                    logger.error(f"❌ {result['name']} download failed")
    
    # Generate and display report
    report = checker.generate_report(results, test_data_results)
    print(report)
    
    # Save results if requested
    if args.save_results:
        checker.save_results(results)
    
    # Exit with appropriate code
    missing_count = sum(1 for r in results.values() if r.get('status') == 'missing')
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing datasets")
        sys.exit(1)
    else:
        logger.info("All datasets are properly installed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
