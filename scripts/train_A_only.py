import numpy as np
import pickle
import os
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from datetime import datetime

print("=" * 60)
print("🤟 TRAINING FOR LETTER 'A' ONLY")
print("=" * 60)

# ===== CONFIGURATION =====
MODELS_DIR = "models"
PROCESSED_DIR = "dataset/processed_landmarks"
os.makedirs(MODELS_DIR, exist_ok=True)

# Create timestamped folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(MODELS_DIR, f"training_A_{timestamp}")
os.makedirs(SESSION_DIR, exist_ok=True)
print(f"\n📁 Session folder: {SESSION_DIR}")

# ===== STEP 1: LOAD OR CREATE DATA FOR 'A' =====
print("\n📂 Preparing data for letter 'A'...")

X = []
y = []

# Try to load real A data first
a_path = os.path.join(PROCESSED_DIR, "A")
if os.path.exists(a_path):
    a_files = [f for f in os.listdir(a_path) if f.endswith('.npy')]
    print(f"Found {len(a_files)} real A samples")
    
    for f in a_files[:100]:  # Use up to 100 samples
        try:
            landmarks = np.load(os.path.join(a_path, f))
            if len(landmarks) == 63:
                X.append(landmarks)
                y.append(1)  # 1 for A
        except:
            pass

# If not enough real data, create synthetic A data
if len(X) < 50:
    print("Creating synthetic A samples...")
    for i in range(100):
        # Create A-like pattern (closed fist, thumb out)
        a_pattern = np.random.randn(63) * 0.1
        # Make thumb coordinates distinctive (landmarks 1-4 are thumb)
        a_pattern[0:12] = [0.5, 0.5, 0.1,  # wrist
                           0.55, 0.45, 0.1,  # thumb base
                           0.6, 0.4, 0.1,    # thumb middle
                           0.65, 0.35, 0.1]  # thumb tip
        X.append(a_pattern)
        y.append(1)

# Create negative samples (NOT A)
print("Creating negative samples (NOT A)...")
for i in range(200):
    # Random patterns for non-A
    not_a = np.random.randn(63) * 0.2
    X.append(not_a)
    y.append(0)  # 0 for NOT A

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total samples: {len(X)}")
print(f"   A samples: {sum(y)}")
print(f"   Not A samples: {len(y) - sum(y)}")

# ===== STEP 2: PREPARE DATA =====
print("\n🔧 Preparing data...")

# Normalize
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std[std == 0] = 1
X_norm = (X - mean) / std

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)}")
print(f"   Validation: {len(X_val)}")
print(f"   Testing: {len(X_test)}")

# ===== STEP 3: CREATE SIMPLE MODEL =====
print("\n🤖 Creating binary classification model...")

model = keras.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 1 output for binary (A vs NOT A)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===== STEP 4: TRAIN =====
print("\n🎯 Training...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ===== STEP 5: EVALUATE =====
print("\n📊 Evaluating...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# ===== STEP 6: SAVE MODEL =====
print("\n💾 Saving model...")

model.save(os.path.join(SESSION_DIR, 'isl_model.keras'))
np.save(os.path.join(SESSION_DIR, 'mean.npy'), mean)
np.save(os.path.join(SESSION_DIR, 'std.npy'), std)

# Save simple class mapping (0=NOT A, 1=A)
class_mapping = {'NOT_A': 0, 'A': 1}
with open(os.path.join(SESSION_DIR, 'class_mapping.pkl'), 'wb') as f:
    pickle.dump(class_mapping, f)

print(f"✅ Model saved to: {SESSION_DIR}")

# ===== STEP 7: TEST WITH SAMPLE PREDICTIONS =====
print("\n🧪 Testing sample predictions:")
for i in range(5):
    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx].reshape(1, -1)
    true = "A" if y_test[idx] == 1 else "NOT A"
    
    pred_prob = model.predict(sample, verbose=0)[0][0]
    pred = "A" if pred_prob > 0.5 else "NOT A"
    
    print(f"   True: {true:5} | Predicted: {pred:5} | Confidence: {pred_prob if pred_prob>0.5 else 1-pred_prob:.2f}")

print("\n" + "="*60)
print("🎉 Training complete!")
print("="*60)
print(f"\n📁 Model folder: {SESSION_DIR}")
print("\n✅ You can now test with webcam using the app!")