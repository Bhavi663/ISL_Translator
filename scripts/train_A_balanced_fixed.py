import numpy as np
import pickle
import os
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import random

print("=" * 60)
print("🤟 BALANCED TRAINING FOR LETTER 'A'")
print("=" * 60)

# ===== CONFIGURATION =====
MODELS_DIR = "models"
PROCESSED_DIR = "dataset/processed_landmarks"
os.makedirs(MODELS_DIR, exist_ok=True)

# Create timestamped folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(MODELS_DIR, f"training_A_balanced_{timestamp}")
os.makedirs(SESSION_DIR, exist_ok=True)
print(f"\n📁 Session folder: {SESSION_DIR}")

# ===== STEP 1: LOAD REAL A DATA =====
print("\n📂 Loading real A samples...")

X = []
y = []  # 1 for A, 0 for NOT A

# Load real A data
a_path = os.path.join(PROCESSED_DIR, "A")
if os.path.exists(a_path):
    a_files = [f for f in os.listdir(a_path) if f.endswith('.npy')]
    print(f"Found {len(a_files)} real A samples")
    
    for f in a_files[:150]:  # Use up to 150 samples
        try:
            landmarks = np.load(os.path.join(a_path, f))
            if len(landmarks) == 63:
                X.append(landmarks)
                y.append(1)
                print(f"  ✓ Loaded A sample: {f}")
        except Exception as e:
            print(f"  ✗ Error loading {f}: {e}")

# ===== STEP 2: LOAD DIVERSE NEGATIVE SAMPLES =====
print("\n📂 Loading diverse negative samples...")

# List of all possible negative classes (everything except A)
negative_classes = []
for folder in os.listdir(PROCESSED_DIR):
    if folder != "A" and os.path.isdir(os.path.join(PROCESSED_DIR, folder)):
        negative_classes.append(folder)

print(f"Found {len(negative_classes)} negative classes: {negative_classes[:10]}...")

negatives_loaded = 0
for neg_class in negative_classes:
    neg_path = os.path.join(PROCESSED_DIR, neg_class)
    neg_files = [f for f in os.listdir(neg_path) if f.endswith('.npy')]
    
    # Take up to 20 samples from each negative class
    for f in neg_files[:20]:
        try:
            landmarks = np.load(os.path.join(neg_path, f))
            if len(landmarks) == 63:
                X.append(landmarks)
                y.append(0)
                negatives_loaded += 1
                if negatives_loaded % 50 == 0:
                    print(f"  Loaded {negatives_loaded} negative samples...")
        except:
            pass

print(f"✅ Loaded {negatives_loaded} negative samples from {len(negative_classes)} classes")

# ===== STEP 3: CREATE SYNTHETIC VARIATIONS =====
print("\n🔄 Creating synthetic variations...")

# Function to add noise to landmarks
def add_noise(landmarks, noise_level=0.05):
    return landmarks + np.random.randn(63) * noise_level

# Function to rotate hand slightly
def rotate_landmarks(landmarks, angle_deg=10):
    angle = np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Reshape to 21 points x 3 coordinates
    points = landmarks.reshape(21, 3)
    rotated = np.dot(points, rotation_matrix)
    return rotated.flatten()

# Function to scale hand
def scale_landmarks(landmarks, scale_factor=0.9):
    points = landmarks.reshape(21, 3)
    scaled = points * scale_factor
    return scaled.flatten()

# Create augmented A samples
X_a = [x for x, label in zip(X, y) if label == 1]
print(f"Creating variations from {len(X_a)} A samples...")

for a_sample in X_a[:50]:  # Use first 50 A samples
    for _ in range(3):  # Create 3 variations per sample
        # Noisy version
        X.append(add_noise(a_sample, 0.03))
        y.append(1)
        
        # Slightly rotated version
        X.append(rotate_landmarks(a_sample, random.uniform(-15, 15)))
        y.append(1)
        
        # Scaled version
        X.append(scale_landmarks(a_sample, random.uniform(0.85, 1.15)))
        y.append(1)

# Create augmented negative samples
X_neg = [x for x, label in zip(X, y) if label == 0]
print(f"Creating variations from {len(X_neg)} negative samples...")

for neg_sample in random.sample(X_neg, min(100, len(X_neg))):
    for _ in range(2):  # Create 2 variations per sample
        X.append(add_noise(neg_sample, 0.05))
        y.append(0)

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total samples after augmentation: {len(X)}")
print(f"   A samples (label=1): {sum(y)}")
print(f"   NOT A samples (label=0): {len(y) - sum(y)}")

# ===== STEP 4: PREPARE DATA =====
print("\n🔧 Preparing data...")

# Normalize
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std[std == 0] = 1
X_norm = (X - mean) / std

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)} (A: {sum(y_train)}, NOT A: {len(y_train)-sum(y_train)})")
print(f"   Validation: {len(X_val)} (A: {sum(y_val)}, NOT A: {len(y_val)-sum(y_val)})")
print(f"   Testing: {len(X_test)} (A: {sum(y_test)}, NOT A: {len(y_test)-sum(y_test)})")

# Calculate class weights to handle imbalance
neg_count = len(y_train) - sum(y_train)
pos_count = sum(y_train)
if pos_count > 0 and neg_count > 0:
    weight_for_neg = (1 / neg_count) * (len(y_train) / 2)
    weight_for_pos = (1 / pos_count) * (len(y_train) / 2)
    class_weight = {0: weight_for_neg, 1: weight_for_pos}
    print(f"\n⚖️ Class weights: NOT A: {weight_for_neg:.2f}, A: {weight_for_pos:.2f}")
else:
    class_weight = None
    print("\n⚠️ Cannot calculate class weights, using default")

# ===== STEP 5: CREATE MODEL =====
print("\n🤖 Creating improved model...")

model = keras.Sequential([
    layers.Input(shape=(63,)),
    
    # First block
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    # Second block
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    # Third block
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Fourth block
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Output
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===== STEP 6: TRAIN =====
print("\n🎯 Training...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
]

# Remove ModelCheckpoint that might cause issues
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val),
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# ===== STEP 7: EVALUATE =====
print("\n📊 Evaluating...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# ===== STEP 8: SAVE MODEL =====
print("\n💾 Saving model...")

model.save(os.path.join(SESSION_DIR, 'isl_model.keras'))
np.save(os.path.join(SESSION_DIR, 'mean.npy'), mean)
np.save(os.path.join(SESSION_DIR, 'std.npy'), std)

# Save class mapping
class_mapping = {'NOT_A': 0, 'A': 1}
with open(os.path.join(SESSION_DIR, 'class_mapping.pkl'), 'wb') as f:
    pickle.dump(class_mapping, f)

print(f"✅ Model saved to: {SESSION_DIR}")

# ===== STEP 9: TEST WITH RANDOM SAMPLES =====
print("\n🧪 Testing on random test samples:")
correct = 0
total_tested = 20

for i in range(total_tested):
    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx].reshape(1, -1)
    true_label = "A" if y_test[idx] == 1 else "NOT A"
    
    pred_prob = model.predict(sample, verbose=0)[0][0]
    pred_label = "A" if pred_prob > 0.5 else "NOT A"
    
    if (pred_label == "A" and y_test[idx] == 1) or (pred_label == "NOT A" and y_test[idx] == 0):
        correct += 1
        marker = "✓"
    else:
        marker = "✗"
    
    print(f"   {marker} True: {true_label:6} | Pred: {pred_label:6} | Conf: {pred_prob if pred_label=='A' else 1-pred_prob:.2f}")

print(f"\n   Test accuracy: {correct}/{total_tested} ({correct/total_tested*100:.1f}%)")

print("\n" + "="*60)
print("🎉 Training complete!")
print("="*60)
print(f"\n📁 Model folder: {SESSION_DIR}")