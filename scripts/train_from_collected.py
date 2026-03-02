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
print("🤟 TRAIN FROM COLLECTED DATA")
print("=" * 60)

# ===== CONFIGURATION =====
PROCESSED_DIR = "dataset/processed_landmarks"
MODELS_DIR = "models"

# Create timestamp for this training
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(MODELS_DIR, f"trained_{timestamp}")
os.makedirs(SESSION_DIR, exist_ok=True)

print(f"\n📁 Session folder: {SESSION_DIR}")

# ===== LOAD ALL COLLECTED DATA =====
print("\n📂 Loading collected data...")

X = []
y = []
class_names = []

# Get all class folders
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
                    if len(landmarks) == 63:
                        X.append(landmarks)
                        y.append(class_name)
                except Exception as e:
                    print(f"    Error loading {lf}: {e}")

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total samples loaded: {len(X)}")
print(f"✅ Number of classes: {len(class_names)}")
print(f"✅ Classes: {class_names}")

if len(X) == 0:
    print("❌ No data found! Please collect data first.")
    exit()

# ===== PREPARE DATA =====
print("\n🔧 Preparing data...")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Calculate normalization parameters
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std[std == 0] = 1

# Normalize
X_norm = (X - mean) / std

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)}")
print(f"   Validation: {len(X_val)}")
print(f"   Testing: {len(X_test)}")

# ===== CREATE MODEL =====
print("\n🤖 Creating model...")

num_classes = len(class_names)

model = keras.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
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
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ===== EVALUATE =====
print("\n📊 Evaluating...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# ===== SAVE MODEL =====
print("\n💾 Saving model...")

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

print(f"✅ Model saved to: {SESSION_DIR}")
print(f"✅ Classes: {list(label_encoder.classes_)}")

print("\n" + "="*60)
print("🎉 Training complete!")
print("="*60)
print(f"\n📁 Model folder: {SESSION_DIR}")
print("\n✅ You can now run the web app!")
print("   The app will automatically load this model")