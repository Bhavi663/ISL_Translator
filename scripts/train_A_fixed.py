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
print("🤟 FIXED TRAINING FOR LETTER 'A' ONLY")
print("=" * 60)

# ===== CONFIGURATION =====
MODELS_DIR = "models"
PROCESSED_DIR = "dataset/processed_landmarks"
os.makedirs(MODELS_DIR, exist_ok=True)

# Create timestamped folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(MODELS_DIR, f"training_A_fixed_{timestamp}")
os.makedirs(SESSION_DIR, exist_ok=True)
print(f"\n📁 Session folder: {SESSION_DIR}")

# ===== STEP 1: LOAD OR CREATE DATA FOR 'A' =====
print("\n📂 Preparing data for letter 'A'...")

X = []
y = []  # We'll use 0 for NOT A, 1 for A

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
                print(f"  ✓ Loaded A sample: {f}")
        except Exception as e:
            print(f"  ✗ Error loading {f}: {e}")

# Create synthetic A data if needed
if len([val for val in y if val == 1]) < 50:
    print("\nCreating synthetic A samples...")
    for i in range(100 - len([val for val in y if val == 1])):
        # Create A-like pattern (closed fist, thumb extended)
        a_pattern = np.random.randn(63) * 0.1
        
        # Make thumb coordinates distinctive (landmarks 1-4 are thumb joints)
        # Typical A sign: thumb extended, other fingers curled
        a_pattern[0:3] = [0.5, 0.5, 0.0]      # wrist
        a_pattern[3:6] = [0.55, 0.45, 0.05]   # thumb CMC
        a_pattern[6:9] = [0.6, 0.4, 0.1]      # thumb MCP
        a_pattern[9:12] = [0.65, 0.35, 0.15]  # thumb IP
        a_pattern[12:15] = [0.7, 0.3, 0.2]    # thumb tip
        
        # Other fingers curled (close to palm)
        for j in range(15, 63, 3):
            a_pattern[j:j+3] = [0.5, 0.6, 0.1]
        
        X.append(a_pattern)
        y.append(1)

# Create negative samples (NOT A)
print("\nCreating negative samples (NOT A)...")
negative_classes = ["B", "C", "D", "E", "F", "1", "2", "3", "4", "5"]

for neg_class in negative_classes[:8]:  # Use 8 negative classes
    neg_path = os.path.join(PROCESSED_DIR, neg_class)
    if os.path.exists(neg_path):
        neg_files = [f for f in os.listdir(neg_path) if f.endswith('.npy')]
        for f in neg_files[:15]:  # Use up to 15 per class
            try:
                landmarks = np.load(os.path.join(neg_path, f))
                if len(landmarks) == 63:
                    X.append(landmarks)
                    y.append(0)  # 0 for NOT A
                    print(f"  ✓ Loaded NOT A sample ({neg_class}): {f}")
            except:
                pass
    
    # Add synthetic negative samples
    for i in range(10):
        not_a = np.random.randn(63) * 0.2
        X.append(not_a)
        y.append(0)

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total samples: {len(X)}")
print(f"   A samples (label=1): {sum(y)}")
print(f"   NOT A samples (label=0): {len(y) - sum(y)}")

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

# ===== STEP 3: CREATE BINARY CLASSIFICATION MODEL =====
print("\n🤖 Creating binary classification model...")

model = keras.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Single output for binary
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# ===== STEP 4: TRAIN =====
print("\n🎯 Training...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ===== STEP 5: EVALUATE =====
print("\n📊 Evaluating...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test Precision: {test_precision:.2f}")
print(f"✅ Test Recall: {test_recall:.2f}")

# ===== STEP 6: SAVE MODEL WITH PROPER CLASS MAPPING =====
print("\n💾 Saving model...")

# Save model
model.save(os.path.join(SESSION_DIR, 'isl_model.keras'))

# Save preprocessing data
np.save(os.path.join(SESSION_DIR, 'mean.npy'), mean)
np.save(os.path.join(SESSION_DIR, 'std.npy'), std)

# CRITICAL: Create proper class mapping for binary classification
# For binary: class 0 = NOT A, class 1 = A
class_to_idx = {
    'NOT_A': 0,
    'A': 1
}

with open(os.path.join(SESSION_DIR, 'class_mapping.pkl'), 'wb') as f:
    pickle.dump(class_to_idx, f)

# Also save a readable version
with open(os.path.join(SESSION_DIR, 'classes.txt'), 'w') as f:
    f.write("NOT_A\nA\n")

print(f"✅ Model saved to: {SESSION_DIR}")
print(f"✅ Class mapping: {class_to_idx}")

# ===== STEP 7: TEST WITH SAMPLE PREDICTIONS =====
print("\n🧪 Testing sample predictions:")
correct = 0
for i in range(10):
    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx].reshape(1, -1)
    true_label = "A" if y_test[idx] == 1 else "NOT A"
    
    pred_prob = model.predict(sample, verbose=0)[0][0]
    pred_label = "A" if pred_prob > 0.5 else "NOT A"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
    
    if (pred_label == "A" and y_test[idx] == 1) or (pred_label == "NOT A" and y_test[idx] == 0):
        correct += 1
        marker = "✓"
    else:
        marker = "✗"
    
    print(f"   {marker} True: {true_label:5} | Pred: {pred_label:5} | Conf: {confidence:.2f}")

print(f"\n   Sample accuracy: {correct}/10")

# ===== STEP 8: CREATE INFO FILE =====
with open(os.path.join(SESSION_DIR, 'README.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("LETTER A BINARY CLASSIFIER\n")
    f.write("="*60 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Total Samples: {len(X)}\n")
    f.write(f"A Samples: {sum(y)}\n")
    f.write(f"NOT A Samples: {len(y) - sum(y)}\n\n")
    f.write("Class Mapping:\n")
    f.write("  0 -> NOT_A\n")
    f.write("  1 -> A\n")

print("\n" + "="*60)
print("🎉 Training complete!")
print("="*60)
print(f"\n📁 Model folder: {SESSION_DIR}")
print("\n✅ You can now run the web app!")
print("   The model will predict 'A' or 'NOT_A'")