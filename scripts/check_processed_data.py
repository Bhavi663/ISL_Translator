import os
import glob

print("=" * 60)
print("📊 CHECKING PROCESSED LANDMARKS")
print("=" * 60)

processed_dir = "dataset/processed_landmarks"

if not os.path.exists(processed_dir):
    print("❌ processed_landmarks folder not found!")
    exit()

total_samples = 0
classes_found = []

for class_name in sorted(os.listdir(processed_dir)):
    class_path = os.path.join(processed_dir, class_name)
    if os.path.isdir(class_path):
        npy_files = glob.glob(os.path.join(class_path, "*.npy"))
        count = len(npy_files)
        total_samples += count
        classes_found.append((class_name, count))
        print(f"  {class_name}: {count} samples")

print(f"\n✅ Total classes: {len(classes_found)}")
print(f"✅ Total samples: {total_samples}")

if total_samples == 0:
    print("\n❌ No data found! Please run auto_collect.py first")
else:
    print("\n✅ Data is ready for training!")