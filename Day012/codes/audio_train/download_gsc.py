import os
import tarfile
import requests
from tqdm import tqdm

# --- Configuration ---
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
OUTPUT_DIR = "google_speech_commands_selected"
TARGET_DIRS = ["on", "up", "down", "left", "right", "off", "_background_noise_"]


def download_and_extract(url, target_dirs, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    temp_filename = "speech_commands_temp.tar.gz"

    # Download logic
    if not os.path.exists(temp_filename):
        print(f"[INFO] Downloading dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(temp_filename, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(f.write(data))

    print(f"[INFO] Extracting targeting: {target_dirs}")

    with tarfile.open(temp_filename, "r:gz") as tar:
        members_to_extract = []

        for member in tar.getmembers():
            # 1. Normalize path (removes leading './' if present)
            clean_path = os.path.normpath(member.name)

            # 2. Split path into parts: ('on', 'file.wav') or ('.', 'on', 'file.wav')
            # Filter out empty strings or current-dir dots from the parts
            parts = [p for p in clean_path.split(os.sep) if p not in ('.', '')]

            # 3. Match if the FIRST directory matches our target list
            if parts and parts[0] in target_dirs:
                members_to_extract.append(member)

        if not members_to_extract:
            print("[ERROR] No files found. Listing first 5 items in archive for debug:")
            print(tar.getnames()[:5])
            return

        for member in tqdm(members_to_extract, desc="Extracting"):
            tar.extract(member, path=output_dir)

    print(f"\n[SUCCESS] Extracted {len(members_to_extract)} items to '{output_dir}'")


if __name__ == "__main__":
    download_and_extract(DATASET_URL, TARGET_DIRS, OUTPUT_DIR)