import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import glob

print("=" * 60)
print("🤟 TRAINING FROM PROCESSED LANDMARKS")
print("=" * 60)

# ===== CONFIGURATION =====
PROCESSED_DIR = "dataset/processed_landmarks"
MODELS_DIR = "models"

# Create timestamp for this training
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(MODELS_DIR, f"trained_{timestamp}")
os.makedirs(SESSION_DIR, exist_ok=True)

print(f"\n📁 Session folder: {SESSION_DIR}")

# ===== LOAD ALL PROCESSED LANDMARKS =====
print("\n📂 Loading processed landmarks...")

X = []
y = []
class_names = []

# Get all class folders from processed_landmarks
for class_name in sorted(os.listdir(PROCESSED_DIR)):
    class_path = os.path.join(PROCESSED_DIR, class_name)
    if os.path.isdir(class_path):
        landmark_files = glob.glob(os.path.join(class_path, "*.npy"))
        
        if landmark_files:
            class_names.append(class_name)
            print(f"  Loading {class_name}: {len(landmark_files)} samples")
            
            for lf in landmark_files:
                try:
                    landmarks = np.load(lf)
                    if len(landmarks) == 63:  # 21 landmarks * 3 coordinates
                        X.append(landmarks)
                        y.append(class_name)
                except Exception as e:
                    print(f"    Error loading {lf}: {e}")

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total samples loaded: {len(X)}")
print(f"✅ Feature dimension: {X.shape[1]}")
print(f"✅ Number of classes: {len(class_names)}")
print(f"✅ Classes: {class_names}")

if len(X) == 0:
    print("❌ No data found in processed_landmarks!")
    print("Please collect data first using auto_collect.py")
    exit()

# ===== CHECK CLASS DISTRIBUTION =====
print("\n📊 Class distribution:")
for class_name in class_names:
    count = np.sum(y == class_name)
    percentage = (count / len(X)) * 100
    print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

# ===== PREPARE DATA =====
print("\n🔧 Preparing data...")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Calculate normalization parameters
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std[std == 0] = 1  # Avoid division by zero

# Normalize
X_norm = (X - mean) / std

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
print(f"   Testing: {len(X_test)} samples")

# ===== CREATE MODEL =====
print("\n🤖 Creating model...")

num_classes = len(class_names)

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
    
    # Output layer
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===== TRAIN =====
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
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ===== EVALUATE =====
print("\n📊 Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# ===== SAVE MODEL AND ARTIFACTS =====
print("\n💾 Saving model and artifacts...")

# Save model
model.save(os.path.join(SESSION_DIR, 'isl_model.keras'))

# Save preprocessing data
np.save(os.path.join(SESSION_DIR, 'mean.npy'), mean)
np.save(os.path.join(SESSION_DIR, 'std.npy'), std)

# Save class mapping
class_to_idx = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
with open(os.path.join(SESSION_DIR, 'class_mapping.pkl'), 'wb') as f:
    pickle.dump(class_to_idx, f)

# Save classes list
with open(os.path.join(SESSION_DIR, 'classes.txt'), 'w') as f:
    for cls in label_encoder.classes_:
        f.write(f"{cls}\n")

# Save training info
info = {
    'timestamp': timestamp,
    'num_samples': len(X),
    'num_classes': num_classes,
    'classes': list(label_encoder.classes_),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'test_accuracy': float(test_acc),
    'class_distribution': {cls: int(np.sum(y == cls)) for cls in class_names}
}

import json
with open(os.path.join(SESSION_DIR, 'training_info.json'), 'w') as f:
    json.dump(info, f, indent=2)

print(f"\n✅ Model saved to: {SESSION_DIR}")
print(f"✅ Classes: {list(label_encoder.classes_)}")

# ===== TEST SOME PREDICTIONS =====
print("\n🧪 Testing sample predictions:")
for i in range(5):
    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx].reshape(1, -1)
    true_label = label_encoder.inverse_transform([y_test[idx]])[0]
    
    pred = model.predict(sample, verbose=0)[0]
    pred_idx = np.argmax(pred)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = pred[pred_idx] * 100
    
    marker = "✓" if pred_label == true_label else "✗"
    print(f"   {marker} True: {true_label:10} | Pred: {pred_label:10} | Conf: {confidence:.1f}%")

print("\n" + "="*60)
print("🎉 Training complete!")
print("="*60)
print(f"\n📁 Model saved in: {SESSION_DIR}")
print("\n✅ You can now run the web app:")
print("   python web_app/app.py")