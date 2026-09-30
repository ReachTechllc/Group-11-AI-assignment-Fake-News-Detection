"""
Dataset Downloader for Fake News Detection
Downloads Fake.csv and True.csv from Kaggle into the datasets/ folder
(next to this script, which is where app.py looks for them).

SETUP:
1. Get Kaggle API credentials: https://www.kaggle.com/settings/account
   -> "Create New API Token" downloads kaggle.json
2. Put kaggle.json in ~/.kaggle/  (Windows: C:\\Users\\YOUR_NAME\\.kaggle\\)
3. Run: python dataset.py

Alternative: set KAGGLE_USERNAME and KAGGLE_KEY environment variables
instead of using kaggle.json.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
DATASET = "jainpooja/fake-news-detection"
REQUIRED = ["Fake.csv", "True.csv"]


def ensure_kaggle_installed():
    """Install the kaggle package only if it's missing"""
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        pass
    except Exception:
        # kaggle raises on import when credentials are missing;
        # that means the package IS installed, so let the credential check handle it
        return True

    print("Installing kaggle package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
        return True
    except subprocess.CalledProcessError:
        print("Failed to install the kaggle package. Try: pip install kaggle")
        return False


def credentials_available():
    """Check for kaggle.json or environment variables"""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("Using Kaggle credentials from environment variables")
        return True

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        print(f"Found kaggle.json at {kaggle_json}")
        try:
            kaggle_json.chmod(0o600)  # Kaggle warns if the key file is readable by others
        except Exception:
            pass
        return True

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nkaggle.json not found at {kaggle_json}")
    print("\nHOW TO GET IT:")
    print("1. Go to https://www.kaggle.com/settings/account")
    print("2. Scroll to the 'API' section and click 'Create New API Token'")
    print(f"3. Move the downloaded kaggle.json into: {kaggle_dir}")
    print("4. Run this script again")
    return False


def already_downloaded():
    return all((DATA_DIR / name).exists() for name in REQUIRED)


def find_file_ci(root, name):
    """Find a file anywhere under root, ignoring letter case"""
    for path in Path(root).rglob("*"):
        if path.is_file() and path.name.lower() == name.lower():
            return path
    return None


def download_dataset():
    """Download and unzip the dataset, then make sure the CSVs end up in datasets/"""
    print(f"\nDownloading {DATASET} (this may take a few minutes)...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=True)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check that kaggle.json is in ~/.kaggle/ and is valid")
        print("2. Check your internet connection")
        print("3. Or download manually from https://www.kaggle.com/datasets/" + DATASET)
        print(f"   and place Fake.csv and True.csv in: {DATA_DIR}")
        return False

    # Fix up the layout if the files were extracted into a subfolder or with different casing
    for name in REQUIRED:
        target = DATA_DIR / name
        if not target.exists():
            found = find_file_ci(DATA_DIR, name)
            if found:
                shutil.move(str(found), str(target))
    return True


def verify_files():
    print("\nVerifying files...")
    ok = True
    total = 0.0
    for name in REQUIRED:
        path = DATA_DIR / name
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            total += size
            print(f"  OK  {name} ({size:.2f} MB)")
        else:
            print(f"  MISSING  {name}")
            ok = False
    if ok:
        print(f"  Total: {total:.2f} MB")
    return ok


def main():
    print("=" * 60)
    print("  FAKE NEWS DETECTION - KAGGLE DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Saving to: {DATA_DIR}")

    if already_downloaded():
        print("\nFake.csv and True.csv already exist. Nothing to download.")
        verify_files()
        return

    if not credentials_available():
        sys.exit(1)

    if not ensure_kaggle_installed():
        sys.exit(1)

    if not download_dataset():
        sys.exit(1)

    if not verify_files():
        print(f"\nThe download finished but the CSV files are not in {DATA_DIR}.")
        print("Check that folder and move Fake.csv / True.csv into it manually.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  SUCCESS! Dataset is ready.")
    print("  Now run:  python app.py")
    print("  (the model trains automatically on first start)")
    print("=" * 60)


if __name__ == "__main__":
    main()