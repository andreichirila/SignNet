#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# Configuration
BATCH_SIZE_MB = 1000  # 1 GB
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
    print(f"\n{'='*80}")
    print(f"Committing batch {batch_num} ({len(files)} files, {total_size_mb:.2f} MB)...")
    print(f"{'='*80}")
    
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
    
    print(f"  Pushing to remote...")
    subprocess.run(
        ['git', 'push'],
        check=True
    )
    
    print(f"✓ Batch {batch_num} committed and pushed successfully\n")

def main():
    print("Starting batch commit process...")
    print(f"Batch size limit: {BATCH_SIZE_MB} MB")
    
    files = get_untracked_and_modified_files()
    
    if not files:
        print("No files to commit")
        return
    
    print(f"Found {len(files)} files to commit")
    
    current_batch = []
    current_batch_size = 0
    batch_number = 1
    
    for file_path in files:
        file_size = os.path.getsize(file_path)
        
        # Check if adding this file would exceed the limit
        would_exceed = current_batch_size + file_size > BATCH_SIZE_BYTES
        
        if would_exceed and current_batch:
            # Commit current batch BEFORE adding the new file
            print(f"\nBatch size would exceed limit ({(current_batch_size + file_size) / (1024**2):.2f} MB)")
            commit_and_push_batch(current_batch, batch_number)
            
            # Start new batch with current file
            current_batch = [file_path]
            current_batch_size = file_size
            batch_number += 1
        else:
            # Add file to current batch
            current_batch.append(file_path)
            current_batch_size += file_size
            
            if batch_number == 1 or len(current_batch) % 10 == 0:
                print(f"  Batch {batch_number}: {len(current_batch)} files, "
                      f"{current_batch_size / (1024**2):.2f} MB")
    
    # Commit remaining files
    if current_batch:
        print(f"\nCommitting final batch...")
        commit_and_push_batch(current_batch, batch_number)
    
    print("="*80)
    print("All batches committed and pushed successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
