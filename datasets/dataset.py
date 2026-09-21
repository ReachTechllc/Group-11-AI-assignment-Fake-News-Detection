"""
Dataset Downloader for Fake News Detection
Automatically downloads Fake.csv and True.csv from Kaggle

SETUP:
1. Get Kaggle API credentials: https://www.kaggle.com/settings/account
2. Download kaggle.json and place in ~/.kaggle/ (created automatically)
3. Run this script: python dataset.py
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

def install_kaggle():
    """Install kaggle package if not already installed"""
    print("📦 Installing kaggle package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
        print("✅ kaggle package installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install kaggle package")
        return False

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("\n🔐 Setting up Kaggle API credentials...")
    
    # Determine kaggle config path
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # Check if kaggle.json exists
    if kaggle_json.exists():
        print(f"✅ Found kaggle.json at {kaggle_json}")
        return True
    else:
        print(f"\n⚠️  kaggle.json not found at {kaggle_json}")
        print("\nHOW TO GET YOUR KAGGLE CREDENTIALS:")
        print("1. Go to: https://www.kaggle.com/settings/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. A file 'kaggle.json' will download")
        print(f"5. Move it to: {kaggle_dir}/")
        print("   (Folder will be created automatically)")
        print("\n💡 TIP: You can also paste your credentials here and I'll create the file for you")
        
        # Try to create directory and ask for manual setup
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✅ Created {kaggle_dir} directory")
        print("Please download kaggle.json and place it there, then run this script again.")
        
        return False

def download_dataset():
    """Download the fake news detection dataset from Kaggle"""
    print("\n📥 Downloading dataset from Kaggle...")
    print("   This may take a few minutes depending on your internet speed...")
    
    try:
        # Import kaggle here (after installation)
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Create API instance
        api = KaggleApi()
        api.authenticate()
        
        # Get current directory
        current_dir = os.getcwd()
        
        # Download dataset
        print("   Downloading: jainpooja/fake-news-detection")
        api.dataset_download_files(
            'jainpooja/fake-news-detection',
            path=current_dir,
            unzip=True
        )
        
        print("✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure kaggle.json is in ~/.kaggle/")
        print("2. Check your internet connection")
        print("3. Visit: https://www.kaggle.com/datasets/jainpooja/fake-news-detection")
        print("   and download manually if needed")
        return False

def verify_files():
    """Verify that CSV files were downloaded"""
    print("\n🔍 Verifying files...")
    
    fake_exists = os.path.exists("Fake.csv")
    true_exists = os.path.exists("True.csv")
    
    if fake_exists and true_exists:
        print("✅ Fake.csv found")
        print("✅ True.csv found")
        
        # Check file sizes
        fake_size = os.path.getsize("Fake.csv") / (1024 * 1024)  # MB
        true_size = os.path.getsize("True.csv") / (1024 * 1024)  # MB
        
        print(f"   Fake.csv: {fake_size:.2f} MB")
        print(f"   True.csv: {true_size:.2f} MB")
        print(f"   Total: {fake_size + true_size:.2f} MB")
        
        return True
    else:
        if not fake_exists:
            print("❌ Fake.csv not found")
        if not true_exists:
            print("❌ True.csv not found")
        return False

def main():
    """Main function to orchestrate the download"""
    print("\n" + "="*60)
    print("  FAKE NEWS DETECTION - KAGGLE DATASET DOWNLOADER")
    print("="*60)
    
    # Step 1: Install kaggle
    if not install_kaggle():
        sys.exit(1)
    
    # Step 2: Setup Kaggle API
    if not setup_kaggle_api():
        print("\n⚠️  Please setup Kaggle credentials first!")
        print("   Then run this script again.")
        sys.exit(1)
    
    # Step 3: Download dataset
    if not download_dataset():
        sys.exit(1)
    
    # Step 4: Verify files
    if not verify_files():
        sys.exit(1)
    
    print("\n" + "="*60)
    print("  ✅ SUCCESS! Dataset is ready to use!")
    print("="*60)
    print("\nYou can now run:")
    print("  python app.py")
    print("\nTo train the model and start the chatbot!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
