from huggingface_hub import hf_hub_download, list_repo_files
import os
import time

def download_with_retry(repo_id, filename, local_dir, max_retries=3):
    """Download a file with retries."""
    for attempt in range(max_retries):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying in 2 seconds...")
            time.sleep(2)

def find_model_file(files):
    """Find the ONNX model file in the repository."""
    onnx_files = [f for f in files if f.endswith('.onnx')]
    if not onnx_files:
        raise ValueError("No .onnx files found in repository")
    # Prefer exact match, then first .onnx file
    return next(
        (f for f in onnx_files if f == 'model.onnx'),
        onnx_files[0]
    )

def download_files():
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    repo_id = "answerdotai/ModernBERT-large"
    
    # List all files in the repository
    print("Listing repository files...")
    all_files = list_repo_files(repo_id)
    print(f"Found {len(all_files)} files in repository")
    
    # Find the model file
    model_file = find_model_file(all_files)
    print(f"Found model file: {model_file}")
    
    # Required config files
    config_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    # Download model file first
    print(f"\nDownloading {model_file}...")
    try:
        path = download_with_retry(repo_id, model_file, data_dir)
        if model_file != "model.onnx":
            # Rename to model.onnx if needed
            new_path = os.path.join(data_dir, "model.onnx")
            os.rename(path, new_path)
            path = new_path
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"Downloaded model file: {size_mb:.1f} MB")
    except Exception as e:
        print(f"Error downloading model file: {e}")
        raise

    # Download config files
    for file in config_files:
        print(f"\nDownloading {file}...")
        try:
            path = download_with_retry(repo_id, file, data_dir)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"Downloaded {file}: {size_mb:.1f} MB")
        except Exception as e:
            print(f"Error downloading {file}: {e}")
            raise

if __name__ == "__main__":
    download_files()
