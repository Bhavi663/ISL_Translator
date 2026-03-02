import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image_path):
    """
    Extract hand landmarks from an image.
    Returns a flattened list of 63 values (21 landmarks * 3 coordinates) if hand detected, else None.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    # Take the first detected hand (should be only one)
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        # Normalized coordinates (x,y,z) are already between 0-1
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def process_data(data_dir="data", output_dir="dataset", test_size=0.2, random_state=42):
    """
    Walk through data_dir, extract landmarks for each image, save features and labels.
    Also split into train/test sets and save the splits.
    """
    # Create output directory if it doesn't exist (do this early!)
    os.makedirs(output_dir, exist_ok=True)

    X = []  # features
    y = []  # labels (as strings)
    class_names = []

    # Get class folders (subdirectories in data_dir)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()  # for consistent ordering

    if not classes:
        print("No class folders found in 'data/'. Please run data collection first.")
        return

    print(f"Found classes: {classes}")

    # Process each class
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"\nProcessing class '{class_name}' with {len(image_files)} images...")

        successful = 0
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            landmarks = extract_landmarks(img_path)
            if landmarks is not None:
                X.append(landmarks)
                y.append(class_name)  # keep as string; we'll encode later
                successful += 1
            else:
                print(f"  No hand detected in {img_file}, skipping.")

        print(f"  Successfully extracted landmarks from {successful} images.")

    if len(X) == 0:
        print("No landmarks extracted. Exiting.")
        return

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    # For labels, we need to map class names to integers
    unique_classes = sorted(set(y))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    y_int = np.array([class_to_idx[label] for label in y], dtype=np.int32)

    # Save the class mapping for later use during inference
    with open(os.path.join(output_dir, "class_mapping.pkl"), "wb") as f:
        pickle.dump(class_to_idx, f)
    print(f"\nClass mapping saved to {output_dir}/class_mapping.pkl")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=test_size, random_state=random_state, stratify=y_int
    )

    # Save datasets
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print(f"\nDataset saved in '{output_dir}/'")
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {len(unique_classes)}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    process_data()