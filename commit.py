#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# Configuration
BATCH_SIZE_MB = 1500  # 1 GB
BATCH_SIZE_BYTES = BATCH_SIZE_MB * 1024 * 1024
COMMIT_MESSAGE = "Batch commit"
FILES_PER_ADD = 50  # Add files in groups to avoid command line length limits

def get_untracked_and_modified_files():
    """Get list of files that need to be committed."""
    result = subprocess.run(
        ['git', 'status', '--porcelain'],
        capture_output=True,
        text=True
    )
    files = []
    for line in result.stdout.strip().split('\n'):
        if line:
            # Extract filename (second column)
            file_path = line[3:].strip()
            if os.path.isfile(file_path):
                files.append(file_path)
    return files

def commit_and_push_batch(files, batch_num):
    """Commit and push a batch of files."""
    if not files:
        return
    
    total_size_mb = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
    print(f"Committing batch {batch_num} ({len(files)} files, {total_size_mb:.2f} MB)...")
    
    # Add files in smaller groups to avoid command line length limits
    for i in range(0, len(files), FILES_PER_ADD):
        file_group = files[i:i + FILES_PER_ADD]
        print(f"  Adding files {i+1} to {min(i+FILES_PER_ADD, len(files))}...")
        subprocess.run(['git', 'add'] + file_group, check=True)
    
    print(f"  Creating commit...")
    subprocess.run(
        ['git', 'commit', '-m', f'{COMMIT_MESSAGE} {batch_num}'],
        check=True
    )
    
    print(f"Batch {batch_num} committed successfully")

def main():
    print("Starting batch commit process...")
    
    files = get_untracked_and_modified_files()
    
    if not files:
        print("No files to commit")
        return
    
    current_batch = []
    current_batch_size = 0
    batch_number = 1
    
    for file_path in files:
        file_size = os.path.getsize(file_path)
        
        # If adding this file exceeds limit, commit current batch
        if current_batch_size + file_size > BATCH_SIZE_BYTES and current_batch:
            commit_and_push_batch(current_batch, batch_number)
            current_batch = []
            current_batch_size = 0
            batch_number += 1
        
        current_batch.append(file_path)
        current_batch_size += file_size
    
    # Commit remaining files
    if current_batch:
        commit_and_push_batch(current_batch, batch_number)
    
    print("All batches committed and pushed successfully!")

if __name__ == '__main__':
    main()
