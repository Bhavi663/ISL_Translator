import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import cv2
import mediapipe as mp
from datetime import datetime
import shutil

print("=" * 60)
print("🤟 SIMPLE ISL MODEL TRAINING - TEST WITH LETTER 'A'")
print("=" * 60)

# ===== CONFIGURATION =====
BASE_MODELS_DIR = "models"
PROCESSED_DIR = "dataset/processed_landmarks"

# Create timestamp for this training session
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = os.path.join(BASE_MODELS_DIR, f"training_{timestamp}")
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"\n📁 Creating new training folder: {MODELS_DIR}")

# ===== STEP 1: LOAD LETTER A DATA =====
print("\n📂 Loading data for letter 'A'...")

X = []
y = []

# Path for letter A
a_path = os.path.join(PROCESSED_DIR, "A")

if os.path.exists(a_path):
    landmark_files = [f for f in os.listdir(a_path) if f.endswith('.npy')]
    print(f"Found {len(landmark_files)} samples for letter A")
    
    # Load all A samples (limit to 200 for balance)
    for lf in landmark_files[:200]:
        try:
            landmarks = np.load(os.path.join(a_path, lf))
            if len(landmarks) == 63:
                X.append(landmarks)
                y.append("A")
        except Exception as e:
            print(f"Error loading {lf}: {e}")
else:
    print("❌ No data found for letter A!")
    print("\nPlease run data collection first:")
    print("python scripts/collect_data.py")
    exit()

# ===== STEP 2: CREATE NEGATIVE SAMPLES =====
print("\n🔄 Creating negative samples (non-A signs)...")

# Add some random noise as negative samples
negative_classes = ["B", "C", "D", "E", "F", "G", "H", "1", "2", "3", "4", "5"]
samples_per_negative = 30  # Reduced to balance with A samples

for neg_class in negative_classes:
    for i in range(samples_per_negative):
        # Create random landmarks with different patterns
        dummy_landmarks = np.random.randn(63) * 0.15
        
        # Add class-specific patterns to make them distinguishable
        if neg_class == "B":
            dummy_landmarks[0:10] += 0.3
            dummy_landmarks[10:20] -= 0.1
        elif neg_class == "C":
            dummy_landmarks[10:20] += 0.3
            dummy_landmarks[20:30] -= 0.1
        elif neg_class == "D":
            dummy_landmarks[20:30] += 0.3
            dummy_landmarks[30:40] -= 0.1
        elif neg_class == "E":
            dummy_landmarks[30:40] += 0.3
        elif neg_class in ["1", "2", "3"]:
            dummy_landmarks[40:50] += 0.3
        
        X.append(dummy_landmarks)
        y.append(neg_class)

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total samples: {len(X)}")
print(f"✅ Feature dimension: {X.shape[1]}")
print(f"✅ Classes: {np.unique(y)}")

# Count samples per class
unique, counts = np.unique(y, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  {cls}: {count} samples")

# ===== STEP 3: PREPARE DATA =====
print("\n🔧 Preparing data for training...")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Classes found: {label_encoder.classes_}")
print(f"Number of classes: {len(label_encoder.classes_)}")

# Calculate normalization parameters
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std[std == 0] = 1

# Normalize
X_normalized = (X - mean) / std

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_encoded, 
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"\n📊 Data split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Testing: {len(X_test)} samples")

# ===== STEP 4: CREATE MODEL =====
print("\n🤖 Creating model...")

num_classes = len(label_encoder.classes_)

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
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Save model architecture
with open(os.path.join(MODELS_DIR, 'model_architecture.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# ===== STEP 5: TRAIN MODEL =====
print("\n🎯 Training model...")

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
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

start_time = time.time()

history = model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time:.2f} seconds")

# ===== STEP 6: EVALUATE =====
print("\n📊 Evaluating model...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy*100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# ===== STEP 7: SAVE MODEL AND ARTIFACTS =====
print("\n💾 Saving model and artifacts...")

# Save final model
model_path = os.path.join(MODELS_DIR, 'isl_model.keras')
model.save(model_path)
print(f"✓ Model saved to: {model_path}")

# Save preprocessing data
np.save(os.path.join(MODELS_DIR, 'mean.npy'), mean)
np.save(os.path.join(MODELS_DIR, 'std.npy'), std)
print(f"✓ Preprocessing data saved")

# Save class mapping
class_to_idx = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
with open(os.path.join(MODELS_DIR, 'class_mapping.pkl'), 'wb') as f:
    pickle.dump(class_to_idx, f)
print(f"✓ Class mapping saved")

# Save class list
with open(os.path.join(MODELS_DIR, 'classes.txt'), 'w') as f:
    for cls in label_encoder.classes_:
        f.write(f"{cls}\n")

# Save training configuration
config = {
    'timestamp': timestamp,
    'num_samples': len(X),
    'num_classes': num_classes,
    'classes': list(label_encoder.classes_),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'training_time': training_time,
    'model_architecture': 'Dense(256)->Dense(128)->Dense(64)->Dense(32)'
}

import json
with open(os.path.join(MODELS_DIR, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print(f"✓ Training config saved")

# ===== STEP 8: PLOT RESULTS =====
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title(f'Model Accuracy\nTest: {test_accuracy*100:.2f}%')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# Class distribution
classes = list(label_encoder.classes_)
counts = [np.sum(y == cls) for cls in classes]
plt.bar(range(len(classes)), counts, color='skyblue', alpha=0.7)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(len(classes)), classes, rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.show()

# ===== STEP 9: CREATE INFO FILE =====
with open(os.path.join(MODELS_DIR, 'README.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("ISL MODEL TRAINING SESSION\n")
    f.write("="*60 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    f.write(f"Number of Classes: {num_classes}\n")
    f.write(f"Classes: {', '.join(label_encoder.classes_)}\n\n")
    f.write("File Descriptions:\n")
    f.write("  - isl_model.keras: Trained model file\n")
    f.write("  - best_model.keras: Best checkpoint during training\n")
    f.write("  - mean.npy: Mean values for normalization\n")
    f.write("  - std.npy: Standard deviation for normalization\n")
    f.write("  - class_mapping.pkl: Class to index mapping\n")
    f.write("  - classes.txt: List of all classes\n")
    f.write("  - training_history.png: Training plots\n")
    f.write("  - training_config.json: Training configuration\n")
    f.write("  - model_architecture.txt: Model architecture summary\n")

print(f"\n✓ Info file saved")

# ===== STEP 10: TEST PREDICTIONS =====
print("\n" + "="*60)
print("🧪 Testing predictions on sample data")
print("="*60)

# Test on a few random samples from test set
for i in range(min(5, len(X_test))):
    sample = X_test[i].reshape(1, -1)
    true_label = label_encoder.inverse_transform([y_test[i]])[0]
    
    pred = model.predict(sample, verbose=0)[0]
    pred_idx = np.argmax(pred)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = pred[pred_idx] * 100
    
    print(f"\nSample {i+1}:")
    print(f"  True: {true_label}")
    print(f"  Predicted: {pred_label} ({confidence:.1f}%)")
    
    # Show top 3 predictions
    top_3_idx = np.argsort(pred)[-3:][::-1]
    print("  Top 3 predictions:")
    for j, idx in enumerate(top_3_idx):
        cls = label_encoder.inverse_transform([idx])[0]
        conf = pred[idx] * 100
        print(f"    {j+1}. {cls}: {conf:.1f}%")

# ===== STEP 11: TEST WITH WEBCAM =====
print("\n" + "="*60)
print("📸 Testing model with webcam (press 'q' to quit)")
print("="*60)

def test_with_webcam(model, mean, std, label_encoder, model_dir):
    """Test the model with live webcam feed."""
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n🎥 Webcam started. Show signs to test!")
    print("Classes:", list(label_encoder.classes_))
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    
    # For smoothing predictions
    prediction_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )
                
                # Extract landmarks for prediction
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                landmarks = np.array(landmarks).reshape(1, -1)
                landmarks_norm = (landmarks - mean) / std
                
                # Make prediction
                pred = model.predict(landmarks_norm, verbose=0)[0]
                idx = np.argmax(pred)
                confidence = pred[idx]
                predicted_class = label_encoder.inverse_transform([idx])[0]
                
                # Smooth predictions
                prediction_history.append(predicted_class)
                if len(prediction_history) > 5:
                    prediction_history.pop(0)
                
                # Get most common prediction in history
                if prediction_history:
                    from collections import Counter
                    most_common = Counter(prediction_history).most_common(1)[0][0]
                    
                    # Display prediction
                    if confidence > 0.7:
                        color = (0, 255, 0)
                        text = f"Sign: {most_common} ({confidence*100:.1f}%)"
                    elif confidence > 0.4:
                        color = (0, 165, 255)
                        text = f"Maybe: {most_common}? ({confidence*100:.1f}%)"
                    else:
                        color = (0, 0, 255)
                        text = f"Uncertain: {most_common}? ({confidence*100:.1f}%)"
                    
                    cv2.putText(frame, text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Show top 3 predictions
                    top_3_idx = np.argsort(pred)[-3:][::-1]
                    y_pos = 70
                    for i, tidx in enumerate(top_3_idx):
                        class_name = label_encoder.inverse_transform([tidx])[0]
                        conf = pred[tidx] * 100
                        cv2.putText(frame, f"{i+1}. {class_name}: {conf:.1f}%", 
                                   (10, y_pos + i*25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add instructions
        cv2.putText(frame, f"Model: {os.path.basename(model_dir)}", (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(f'ISL Model Test - {os.path.basename(model_dir)}', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            screenshot_path = os.path.join(model_dir, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(screenshot_path, frame)
            print(f"  Screenshot saved: {screenshot_path}")
    
    cap.release()
    cv2.destroyAllWindows()

# Ask if user wants to test with webcam
test_choice = input("\n📸 Test model with webcam? (y/n): ").strip().lower()
if test_choice == 'y':
    test_with_webcam(model, mean, std, label_encoder, MODELS_DIR)

# ===== FINAL SUMMARY =====
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"\n📁 Model saved in: {MODELS_DIR}")
print(f"📊 Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"📋 Classes trained: {list(label_encoder.classes_)}")
print(f"\n📂 Folder contents:")
for file in os.listdir(MODELS_DIR):
    size = os.path.getsize(os.path.join(MODELS_DIR, file)) / 1024  # KB
    print(f"  - {file} ({size:.1f} KB)")

print("\n" + "="*60)
print("✅ To use this model in the web app, you need to:")
print("1. Copy the files from this folder to the main models folder:")
print(f"   copy {MODELS_DIR}\\* models\\")
print("2. Or modify app.py to load from a specific model path")
print("="*60)

# ===== OPTIONAL: COPY TO MAIN MODELS FOLDER =====
copy_choice = input("\n📋 Copy this model to main models folder? (y/n): ").strip().lower()
if copy_choice == 'y':
    # Backup existing models if they exist
    main_model_path = os.path.join(BASE_MODELS_DIR, 'isl_model.keras')
    if os.path.exists(main_model_path):
        backup_dir = os.path.join(BASE_MODELS_DIR, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup all existing model files
        for file in os.listdir(BASE_MODELS_DIR):
            if os.path.isfile(os.path.join(BASE_MODELS_DIR, file)):
                shutil.copy2(
                    os.path.join(BASE_MODELS_DIR, file),
                    os.path.join(backup_dir, file)
                )
        print(f"✓ Existing models backed up to: {backup_dir}")
    
    # Copy new model files to main models folder
    for file in os.listdir(MODELS_DIR):
        if file.endswith(('.keras', '.npy', '.pkl', '.txt')):
            src = os.path.join(MODELS_DIR, file)
            dst = os.path.join(BASE_MODELS_DIR, file)
            shutil.copy2(src, dst)
            print(f"✓ Copied: {file}")
    
    print("\n✅ New model copied to main models folder!")
    print("You can now run the web app with: python run.py")