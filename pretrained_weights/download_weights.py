#pre-trained model weights.

import os
import urllib.request
import sys
import zipfile
import argparse
import hashlib

# URL to the model weights
MODEL_URLS = {
    "figure_eight": "path/to/tia_figure_eight_weights.zip",
    "wikisql": "path/to/tia_wikisql_weights.zip",
    "mturk": "path/to/tia_mturk_weights.zip",
    "all": "path/to/tia_all_weights.zip"
}

# MD5 checksums 
MD5_CHECKSUMS = {
    "tia_figure_eight_weights.zip": "abcdef1234567890abcdef1234567890",
    "tia_wikisql_weights.zip": "1234567890abcdef1234567890abcdef",
    "tia_mturk_weights.zip": "890abcdef1234567890abcdef12345678",
    "tia_all_weights.zip": "567890abcdef1234567890abcdef1234"
}

def calculate_md5(file_path):
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, destination):
    
    print(f"Downloading from {url} to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress reporting
    def report_progress(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            sys.stdout.write(f"\r{percent:.2f}% ({read_so_far} / {total_size} bytes)")
            if read_so_far >= total_size:
                sys.stdout.write("\n")
        else:
            sys.stdout.write(f"\rRead {read_so_far} bytes")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, report_progress)
        print(f"\nDownload completed: {destination}")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def verify_checksum(file_path, expected_md5):
    """Verify the MD5 checksum of a downloaded file."""
    calculated_md5 = calculate_md5(file_path)
    if calculated_md5 == expected_md5:
        print(f"Checksum verified: {calculated_md5}")
        return True
    else:
        print(f"Checksum mismatch: expected {expected_md5}, got {calculated_md5}")
        return False

def extract_zip(zip_path, extract_dir):
     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction completed")

def download_weights(dataset="all", output_dir="pretrained_weights", verify=True, keep_zip=False):
   
    if dataset not in MODEL_URLS:
        print(f"Unknown dataset: {dataset}")
        print(f"Available datasets: {', '.join(MODEL_URLS.keys())}")
        return False
    
    url = MODEL_URLS[dataset]
    zip_filename = url.split("/")[-1]
    zip_path = os.path.join(output_dir, zip_filename)
    
    # Download the weights
    if not download_file(url, zip_path):
        return False
    
    # Verify checksum
    if verify and zip_filename in MD5_CHECKSUMS:
        if not verify_checksum(zip_path, MD5_CHECKSUMS[zip_filename]):
           return False
    
    # Extract the weights
    extract_zip(zip_path, output_dir)
    
  
    if not keep_zip:
        os.remove(zip_path)
       
    
    print(f"Successfully downloaded and extracted weights for {dataset} dataset.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pre-trained weights")
    parser.add_argument("--dataset", type=str, default="all", choices=list(MODEL_URLS.keys()),
                        help="Dataset name (figure_eight, wikisql, mturk, or all)")
    parser.add_argument("--output-dir", type=str, default="pretrained_weights")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip checksum verification")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the zip file after extraction")
    
    args = parser.parse_args()
    
    download_weights(
        dataset=args.dataset,
        output_dir=args.output_dir,
        verify=not args.no_verify,
        keep_zip=args.keep_zip
    )
