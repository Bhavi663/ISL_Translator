import os
import numpy as np
from collections import Counter

def check_dataset():
    """Check dataset statistics."""
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    
    # Check raw images
    raw_dir = "dataset/raw_images"
    if os.path.exists(raw_dir):
        raw_classes = [d for d in os.listdir(raw_dir) 
                      if os.path.isdir(os.path.join(raw_dir, d))]
        
        print(f"\n📸 Raw Images:")
        print(f"  Classes: {len(raw_classes)}")
        
        total_raw = 0
        for cls in sorted(raw_classes):
            cls_dir = os.path.join(raw_dir, cls)
            count = len([f for f in os.listdir(cls_dir) 
                        if f.endswith(('.jpg', '.png', '.jpeg'))])
            total_raw += count
            print(f"  {cls}: {count} images")
        
        print(f"\n  Total raw images: {total_raw}")
    
    # Check processed landmarks
    processed_dir = "dataset/processed_landmarks"
    if os.path.exists(processed_dir):
        processed_classes = [d for d in os.listdir(processed_dir) 
                            if os.path.isdir(os.path.join(processed_dir, d))]
        
        print(f"\n🖐️ Processed Landmarks:")
        print(f"  Classes: {len(processed_classes)}")
        
        total_processed = 0
        for cls in sorted(processed_classes):
            cls_dir = os.path.join(processed_dir, cls)
            count = len([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
            total_processed += count
            print(f"  {cls}: {count} landmarks")
        
        print(f"\n  Total processed landmarks: {total_processed}")
        
        # Check if any classes need more data
        print("\n📈 Recommendations:")
        min_samples = 50
        need_more = []
        for cls in processed_classes:
            cls_dir = os.path.join(processed_dir, cls)
            count = len([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
            if count < min_samples:
                need_more.append((cls, count))
        
        if need_more:
            print("  Classes needing more data (<50 samples):")
            for cls, count in sorted(need_more):
                print(f"    {cls}: {count} samples (need {min_samples-count} more)")
        else:
            print("  All classes have sufficient data (>=50 samples)")

if __name__ == "__main__":
    check_dataset()