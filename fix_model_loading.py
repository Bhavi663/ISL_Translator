import os
import glob
import pickle
import numpy as np

print("="*60)
print("🔧 FIXING MODEL LOADING")
print("="*60)

# Check models directory
models_dir = "models"
if not os.path.exists(models_dir):
    print("❌ Models directory not found!")
    exit()

# Find trained_* folders
trained_folders = glob.glob(os.path.join(models_dir, "trained_*"))
print(f"\n📁 Found {len(trained_folders)} trained folders:")

if not trained_folders:
    print("❌ No trained_* folders found!")
    exit()

latest_folder = sorted(trained_folders)[-1]
print(f"✅ Latest folder: {os.path.basename(latest_folder)}")

# Check files in the latest folder
model_file = os.path.join(latest_folder, "isl_model.keras")
mean_file = os.path.join(latest_folder, "mean.npy")
std_file = os.path.join(latest_folder, "std.npy")
mapping_file = os.path.join(latest_folder, "class_mapping.pkl")

print(f"\n📄 Checking files in {os.path.basename(latest_folder)}:")
print(f"   isl_model.keras exists: {os.path.exists(model_file)}")
print(f"   mean.npy exists: {os.path.exists(mean_file)}")
print(f"   std.npy exists: {os.path.exists(std_file)}")
print(f"   class_mapping.pkl exists: {os.path.exists(mapping_file)}")

# Load and display class mapping
if os.path.exists(mapping_file):
    with open(mapping_file, 'rb') as f:
        class_mapping = pickle.load(f)
    print(f"\n📋 Class mapping: {class_mapping}")
    print(f"📊 Number of classes: {len(class_mapping)}")

# Create a simple test to verify
print("\n" + "="*60)
print("✅ To fix the app:")
print("1. Make sure your app.py has the corrected find_latest_model() function")
print("2. The function should look for 'trained_*' folders, not 'training_*'")
print("3. Restart your Flask app")
print("="*60)