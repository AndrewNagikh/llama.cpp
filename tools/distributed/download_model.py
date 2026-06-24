#!/usr/bin/env python3
"""
Simple HuggingFace model downloader for distributed-llm.
Downloads GGUF models to local model store directory.
"""

import os
import sys
import json
import time
from pathlib import Path
import argparse

def download_with_hf_hub(repo_id: str, filename: str, target_path: str) -> bool:
    """Download using huggingface_hub library."""
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"Downloading {repo_id}/{filename}...")
        
        # Create parent directory
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Download file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(target_path),
            local_dir_use_symlinks=False
        )
        
        # Move to expected location if needed
        if downloaded_path != target_path:
            import shutil
            shutil.move(downloaded_path, target_path)
        
        print(f"Downloaded to: {target_path}")
        return True
        
    except ImportError:
        print("huggingface_hub not installed, trying hf CLI...")
        return False
    except Exception as e:
        print(f"Error downloading with huggingface_hub: {e}")
        return False

def download_with_hf_cli(repo_id: str, filename: str, target_path: str) -> bool:
    """Download using hf CLI tool."""
    try:
        import subprocess
        
        print(f"Downloading {repo_id}/{filename} using hf CLI...")
        
        # Create parent directory
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Use hf download command
        result = subprocess.run([
            "hf", "download", repo_id, filename,
            "--local-dir", os.path.dirname(target_path),
            "--local-dir-use-symlinks", "False"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"hf CLI failed: {result.stderr}")
            return False
        
        # Check if file exists
        if os.path.exists(target_path):
            print(f"Downloaded to: {target_path}")
            return True
        else:
            print(f"File not found at expected location: {target_path}")
            return False
            
    except FileNotFoundError:
        print("hf CLI not found in PATH")
        return False
    except Exception as e:
        print(f"Error downloading with hf CLI: {e}")
        return False

def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(path)
    except:
        return 0

def main():
    parser = argparse.ArgumentParser(description="Download GGUF model from HuggingFace")
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID")
    parser.add_argument("--file", required=True, help="GGUF filename")
    parser.add_argument("--output", required=True, help="Target file path")
    parser.add_argument("--progress-file", help="JSON file to write progress updates")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    def write_progress(status: str, progress: float = 0.0, error: str = ""):
        if args.progress_file:
            progress_data = {
                "status": status,
                "progress": progress,
                "error": error,
                "elapsed_ms": int((time.time() - start_time) * 1000)
            }
            try:
                with open(args.progress_file, 'w') as f:
                    json.dump(progress_data, f)
            except:
                pass
    
    write_progress("starting")
    
    # Check if already exists
    if os.path.exists(args.output):
        file_size = get_file_size(args.output)
        if file_size > 0:
            print(f"File already exists: {args.output} ({file_size} bytes)")
            write_progress("ready", 1.0)
            return 0
    
    write_progress("downloading", 0.1)
    
    # Try downloading with different methods
    success = False
    
    # Method 1: huggingface_hub library
    if not success:
        success = download_with_hf_hub(args.repo, args.file, args.output)
    
    # Method 2: hf CLI
    if not success:
        success = download_with_hf_cli(args.repo, args.file, args.output)
    
    if success:
        file_size = get_file_size(args.output)
        print(f"Download complete: {file_size} bytes")
        write_progress("ready", 1.0)
        return 0
    else:
        error_msg = "All download methods failed. Please install huggingface_hub or hf CLI."
        print(f"Error: {error_msg}")
        write_progress("error", 0.0, error_msg)
        return 1

if __name__ == "__main__":
    sys.exit(main())