import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load the preprocessed data
print("Loading dataset...")
X_train = np.load("dataset/X_train.npy")
X_test = np.load("dataset/X_test.npy")
y_train = np.load("dataset/y_train.npy")
y_test = np.load("dataset/y_test.npy")

# Load class mapping (to know the names)
with open("dataset/class_mapping.pkl", "rb") as f:
    class_to_idx = pickle.load(f)
# Create reverse mapping for later
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Number of classes: {num_classes}")
print(f"Feature dimension: {X_train.shape[1]}")

# Normalize input features? Landmarks are already normalized between 0-1, but we can still standardize if needed.
# Optional: Standardize (zero mean, unit variance) – can help training.
# Compute mean and std from training set
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
# Avoid division by zero
std[std == 0] = 1.0
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Save the normalization parameters for later use in real-time inference
np.save("models/mean.npy", mean)
np.save("models/std.npy", std)

# Build the model
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
print("\nTraining started...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,  # use 20% of training data for validation during training
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Generate predictions for test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=[idx_to_class[i] for i in range(num_classes)]))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("models/training_history.png")
plt.show()

# Save the model in the native Keras format
model.save("models/isl_model.keras")
print("\nModel saved to 'models/isl_model.keras'")