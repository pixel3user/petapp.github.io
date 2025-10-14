# Google Drive Download Script Guide

This guide explains how to use the `download_from_gdrive.py` script to download and extract zip files from your Google Drive.

## Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_gdrive.txt
   ```

2. **Google Drive API Setup**:
   - The script uses your existing `client_secret_143419068550-kfibgjq0u505n0gt1bvhmc1kt7i2c71s.apps.googleusercontent.com.json` file
   - This file contains your OAuth2 credentials for Google Drive API access

## Usage Options

### 1. Interactive Mode (Recommended for first use)
```bash
python download_from_gdrive.py
```
This will:
- List all zip files in your Google Drive
- Let you choose which files to download
- Ask if you want to keep or remove zip files after extraction

### 2. List All Zip Files
```bash
python download_from_gdrive.py --list-files
```
Shows all zip files in your Google Drive with their IDs, names, sizes, and modification dates.

### 3. Download All Zip Files
```bash
python download_from_gdrive.py --all-zips
```
Downloads and extracts all zip files from your Google Drive.

### 4. Download Specific File
```bash
python download_from_gdrive.py --file-id YOUR_FILE_ID
```
Downloads a specific file by its Google Drive file ID.

### 5. Keep Zip Files After Extraction
```bash
python download_from_gdrive.py --all-zips --no-cleanup
```
Downloads and extracts all files but keeps the zip files in the root directory.

## Authentication Process

1. **First Run**: The script will open a web browser for OAuth2 authentication
2. **Sign in** to your Google account
3. **Grant permissions** for the script to access your Google Drive
4. **Credentials are saved** in `token.json` for future use

## What the Script Does

1. **Authenticates** with Google Drive API using OAuth2
2. **Lists** all zip files in your Google Drive
3. **Downloads** selected zip files to the root directory
4. **Extracts** zip files to the root directory (preserving folder structure)
5. **Optionally removes** zip files after extraction
6. **Logs** all operations to `gdrive_download.log`

## Output Structure

After running the script, your root directory will contain:
```
/teamspace/studios/this_studio/
├── download_from_gdrive.py
├── gdrive_download.log
├── token.json (created after first authentication)
└── [extracted folders from zip files]
```

## Troubleshooting

### Authentication Issues
- Delete `token.json` and run the script again to re-authenticate
- Ensure your `client_secret_*.json` file is valid and not corrupted

### Permission Issues
- Make sure the Google account has access to the files you want to download
- Check that the OAuth2 app has the necessary Drive API permissions

### File Not Found
- Use `--list-files` to see all available zip files
- Verify the file ID is correct if using `--file-id`

### Network Issues
- The script will retry failed downloads automatically
- Check your internet connection and Google Drive API status

## Log Files

The script creates detailed logs in `gdrive_download.log` including:
- Authentication status
- File discovery
- Download progress
- Extraction results
- Error messages

## Security Notes

- The `token.json` file contains sensitive authentication data
- Keep this file secure and don't share it
- The script only requests read-only access to your Google Drive

## Example Workflow

1. **First time setup**:
   ```bash
   pip install -r requirements_gdrive.txt
   python download_from_gdrive.py --list-files
   ```

2. **Download all zip files**:
   ```bash
   python download_from_gdrive.py --all-zips
   ```

3. **Check results**:
   ```bash
   ls -la /teamspace/studios/this_studio/
   ```

The script is designed to be safe and will not overwrite existing files without warning.







