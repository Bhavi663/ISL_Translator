import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import mediapipe as mp

print("=" * 60)
print("🚀 QUICK ISL MODEL TEST")
print("=" * 60)

# Create synthetic data for testing
print("\n📊 Creating synthetic training data...")

X = []
y = []

# Create 100 samples for A
for i in range(100):
    # Pattern for A (thumb out, fingers curled)
    landmarks = np.random.randn(63) * 0.05
    # Add specific pattern for A
    landmarks[0:3] = [0.5, 0.5, 0.1]  # wrist
    landmarks[3:6] = [0.55, 0.45, 0.1]  # thumb
    landmarks[6:9] = [0.5, 0.4, 0.1]  # index curled
    X.append(landmarks)
    y.append("A")

# Create 50 samples for B
for i in range(50):
    # Pattern for B (all fingers straight)
    landmarks = np.random.randn(63) * 0.05 + 0.2
    landmarks[0:3] = [0.5, 0.5, 0.1]
    landmarks[6:9] = [0.5, 0.3, 0.1]  # index straight
    landmarks[9:12] = [0.55, 0.3, 0.1]  # middle straight
    X.append(landmarks)
    y.append("B")

# Create 50 samples for C
for i in range(50):
    # Pattern for C (curved hand)
    landmarks = np.random.randn(63) * 0.05 + 0.3
    X.append(landmarks)
    y.append("C")

X = np.array(X)
y = np.array(y)

print(f"Created {len(X)} total samples")
print(f"Classes: {set(y)}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std[std == 0] = 1
X_normalized = (X - mean) / std

# Create model
model = keras.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🤖 Training model...")
model.fit(X_normalized, y_encoded, epochs=20, batch_size=16, verbose=1)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/isl_model.keras")
np.save("models/mean.npy", mean)
np.save("models/std.npy", std)

class_to_idx = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
with open("models/class_mapping.pkl", "wb") as f:
    pickle.dump(class_to_idx, f)

print("\n✅ Model saved! You can now run the web app.")
print("\n👉 Run: python run.py")