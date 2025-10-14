#!/usr/bin/env python3
"""
Google Drive Download and Unzip Script
=====================================

This script downloads zip files from Google Drive and extracts them to the root directory.
It uses the Google Drive API v3 with OAuth2 authentication.

Features:
- Authenticate with Google Drive using OAuth2
- List all zip files in your Google Drive
- Download selected zip files
- Extract zip files to the root directory
- Handle errors and provide detailed logging

Usage:
    python download_from_gdrive.py [--file-id FILE_ID] [--list-files] [--all-zips]

Requirements:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

import os
import sys
import json
import zipfile
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gdrive_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Configuration
CLIENT_SECRET_FILE = 'client_secret_143419068550-kfibgjq0u505n0gt1bvhmc1kt7i2c71s.apps.googleusercontent.com.json'
TOKEN_FILE = 'token.json'
ROOT_DIR = '/teamspace/studios/this_studio'


class GoogleDriveDownloader:
    """Google Drive downloader with authentication and file management"""
    
    def __init__(self, client_secret_file: str = CLIENT_SECRET_FILE):
        self.client_secret_file = client_secret_file
        self.service = None
        self.credentials = None
        
    def authenticate(self) -> bool:
        """Authenticate with Google Drive API"""
        logger.info("Starting Google Drive authentication...")
        
        # Load existing credentials
        if os.path.exists(TOKEN_FILE):
            self.credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            logger.info("✓ Loaded existing credentials")
        
        # If there are no valid credentials, get new ones
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                logger.info("Refreshing expired credentials...")
                self.credentials.refresh(Request())
                logger.info("✓ Credentials refreshed")
            else:
                logger.info("Starting OAuth2 flow...")
                if not os.path.exists(self.client_secret_file):
                    logger.error(f"❌ Client secret file not found: {self.client_secret_file}")
                    return False
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secret_file, SCOPES
                )
                self.credentials = flow.run_local_server(port=0)
                logger.info("✓ OAuth2 flow completed")
            
            # Save credentials for next run
            with open(TOKEN_FILE, 'w') as token:
                token.write(self.credentials.to_json())
            logger.info(f"✓ Credentials saved to {TOKEN_FILE}")
        
        # Build the service
        try:
            self.service = build('drive', 'v3', credentials=self.credentials)
            logger.info("✓ Google Drive service initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Drive service: {e}")
            return False
    
    def list_zip_files(self, query: str = "mimeType='application/zip'", folder_id: str = None) -> List[Dict]:
        """List all zip files in Google Drive, optionally in a specific folder"""
        if folder_id:
            logger.info(f"Searching for zip files in folder {folder_id}...")
            query = f"'{folder_id}' in parents and {query}"
        else:
            logger.info("Searching for zip files in Google Drive...")
        
        try:
            results = self.service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, size, createdTime, modifiedTime, parents)",
                orderBy="modifiedTime desc"
            ).execute()
            
            files = results.get('files', [])
            logger.info(f"✓ Found {len(files)} zip files")
            
            return files
            
        except Exception as e:
            logger.error(f"❌ Failed to list files: {e}")
            return []
    
    def get_file_info(self, file_id: str) -> Optional[Dict]:
        """Get detailed information about a specific file"""
        try:
            file_info = self.service.files().get(
                fileId=file_id,
                fields="id, name, size, mimeType, createdTime, modifiedTime"
            ).execute()
            return file_info
        except Exception as e:
            logger.error(f"❌ Failed to get file info for {file_id}: {e}")
            return None
    
    def download_file(self, file_id: str, filename: str) -> bool:
        """Download a file from Google Drive"""
        logger.info(f"Downloading {filename} (ID: {file_id})...")
        
        try:
            # Get file info
            file_info = self.get_file_info(file_id)
            if not file_info:
                return False
            
            file_size = int(file_info.get('size', 0))
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            # Download the file
            request = self.service.files().get_media(fileId=file_id)
            file_handle = io.BytesIO()
            downloader = MediaIoBaseDownload(file_handle, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"Download progress: {progress}%")
            
            # Save to file
            file_path = os.path.join(ROOT_DIR, filename)
            with open(file_path, 'wb') as f:
                f.write(file_handle.getvalue())
            
            logger.info(f"✓ Downloaded {filename} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            return False
    
    def extract_zip(self, zip_path: str) -> bool:
        """Extract zip file to root directory"""
        logger.info(f"Extracting {zip_path}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files in zip
                file_list = zip_ref.namelist()
                logger.info(f"Zip contains {len(file_list)} files/folders")
                
                # Extract all files
                zip_ref.extractall(ROOT_DIR)
                
                # Log extracted folders
                extracted_folders = set()
                for file_path in file_list:
                    if '/' in file_path:
                        folder = file_path.split('/')[0]
                        extracted_folders.add(folder)
                
                if extracted_folders:
                    logger.info(f"✓ Extracted folders: {', '.join(extracted_folders)}")
                else:
                    logger.info("✓ Extracted files to root directory")
                
                return True
                
        except zipfile.BadZipFile:
            logger.error(f"❌ {zip_path} is not a valid zip file")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to extract {zip_path}: {e}")
            return False
    
    def cleanup_zip(self, zip_path: str) -> bool:
        """Remove the downloaded zip file after extraction"""
        try:
            os.remove(zip_path)
            logger.info(f"✓ Cleaned up {zip_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove {zip_path}: {e}")
            return False
    
    def download_and_extract(self, file_id: str, filename: str, cleanup: bool = True) -> bool:
        """Download and extract a zip file"""
        logger.info(f"Processing {filename}...")
        
        # Download the file
        if not self.download_file(file_id, filename):
            return False
        
        # Extract the zip
        zip_path = os.path.join(ROOT_DIR, filename)
        if not self.extract_zip(zip_path):
            return False
        
        # Cleanup if requested
        if cleanup:
            self.cleanup_zip(zip_path)
        
        return True
    
    def download_all_zips(self, cleanup: bool = True, folder_id: str = None) -> int:
        """Download and extract all zip files from Google Drive"""
        if folder_id:
            logger.info(f"Downloading all zip files from folder {folder_id}...")
        else:
            logger.info("Downloading all zip files from Google Drive...")
        
        # Get all zip files
        zip_files = self.list_zip_files(folder_id=folder_id)
        if not zip_files:
            logger.warning("No zip files found in Google Drive")
            return 0
        
        # Process each zip file
        success_count = 0
        for file_info in zip_files:
            file_id = file_info['id']
            filename = file_info['name']
            
            logger.info(f"\n--- Processing {filename} ---")
            
            if self.download_and_extract(file_id, filename, cleanup):
                success_count += 1
            else:
                logger.error(f"Failed to process {filename}")
        
        logger.info(f"\n✓ Successfully processed {success_count}/{len(zip_files)} zip files")
        return success_count


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download and extract zip files from Google Drive")
    parser.add_argument('--file-id', help='Specific file ID to download')
    parser.add_argument('--list-files', action='store_true', help='List all zip files in Google Drive')
    parser.add_argument('--all-zips', action='store_true', help='Download all zip files')
    parser.add_argument('--folder-id', help='Specific folder ID to search in')
    parser.add_argument('--no-cleanup', action='store_true', help='Keep zip files after extraction')
    parser.add_argument('--client-secret', default=CLIENT_SECRET_FILE, help='Path to client secret file')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = GoogleDriveDownloader(args.client_secret)
    
    # Authenticate
    if not downloader.authenticate():
        logger.error("❌ Authentication failed")
        sys.exit(1)
    
    # Handle different modes
    if args.list_files:
        # List all zip files
        zip_files = downloader.list_zip_files(folder_id=args.folder_id)
        if zip_files:
            folder_info = f" in folder {args.folder_id}" if args.folder_id else ""
            print(f"\n📁 Zip files in Google Drive{folder_info}:")
            print("-" * 80)
            for file_info in zip_files:
                size_mb = int(file_info.get('size', 0)) / (1024 * 1024)
                print(f"ID: {file_info['id']}")
                print(f"Name: {file_info['name']}")
                print(f"Size: {size_mb:.2f} MB")
                print(f"Modified: {file_info.get('modifiedTime', 'Unknown')}")
                print("-" * 80)
        else:
            folder_info = f" in folder {args.folder_id}" if args.folder_id else ""
            print(f"No zip files found in Google Drive{folder_info}")
    
    elif args.file_id:
        # Download specific file
        file_info = downloader.get_file_info(args.file_id)
        if not file_info:
            logger.error(f"❌ File with ID {args.file_id} not found")
            sys.exit(1)
        
        filename = file_info['name']
        cleanup = not args.no_cleanup
        
        if downloader.download_and_extract(args.file_id, filename, cleanup):
            logger.info(f"✅ Successfully processed {filename}")
        else:
            logger.error(f"❌ Failed to process {filename}")
            sys.exit(1)
    
    elif args.all_zips:
        # Download all zip files
        cleanup = not args.no_cleanup
        success_count = downloader.download_all_zips(cleanup, folder_id=args.folder_id)
        
        if success_count > 0:
            logger.info(f"✅ Successfully processed {success_count} zip files")
        else:
            logger.error("❌ No zip files were processed successfully")
            sys.exit(1)
    
    else:
        # Default behavior: download all zip files from the specified folder
        if args.folder_id:
            logger.info(f"Downloading all zip files from folder {args.folder_id}...")
            cleanup = not args.no_cleanup
            success_count = downloader.download_all_zips(cleanup, folder_id=args.folder_id)
            
            if success_count > 0:
                logger.info(f"✅ Successfully processed {success_count} zip files")
            else:
                logger.error("❌ No zip files were processed successfully")
                sys.exit(1)
        else:
            # Interactive mode for when no folder is specified
            print("\n🔍 Google Drive Zip File Downloader")
            print("=" * 50)
            
            # List available files
            zip_files = downloader.list_zip_files()
            if not zip_files:
                print("No zip files found in Google Drive")
                sys.exit(0)
            
            print(f"\nFound {len(zip_files)} zip files:")
            for i, file_info in enumerate(zip_files, 1):
                size_mb = int(file_info.get('size', 0)) / (1024 * 1024)
                print(f"{i}. {file_info['name']} ({size_mb:.2f} MB)")
            
            # Get user choice
            while True:
                try:
                    choice = input(f"\nEnter file number (1-{len(zip_files)}) or 'all' for all files: ").strip()
                    
                    if choice.lower() == 'all':
                        cleanup = input("Remove zip files after extraction? (y/n): ").strip().lower() == 'y'
                        success_count = downloader.download_all_zips(cleanup)
                        if success_count > 0:
                            logger.info(f"✅ Successfully processed {success_count} zip files")
                        break
                    
                    file_index = int(choice) - 1
                    if 0 <= file_index < len(zip_files):
                        file_info = zip_files[file_index]
                        cleanup = input("Remove zip file after extraction? (y/n): ").strip().lower() == 'y'
                        
                        if downloader.download_and_extract(file_info['id'], file_info['name'], cleanup):
                            logger.info(f"✅ Successfully processed {file_info['name']}")
                        else:
                            logger.error(f"❌ Failed to process {file_info['name']}")
                        break
                    else:
                        print("Invalid choice. Please try again.")
                
                except ValueError:
                    print("Invalid input. Please enter a number or 'all'.")
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                    sys.exit(0)


if __name__ == "__main__":
    main()
