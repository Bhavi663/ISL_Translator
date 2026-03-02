import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf
import os

print("="*60)
print("🧪 DIRECT MODEL TEST")
print("="*60)

# Find the latest model
models_dir = "models"
trained_folders = [f for f in os.listdir(models_dir) if f.startswith('trained_')]
latest_folder = sorted(trained_folders)[-1]
folder_path = os.path.join(models_dir, latest_folder)

print(f"📁 Testing model: {latest_folder}")

# Load model and files
model = tf.keras.models.load_model(os.path.join(folder_path, "isl_model.keras"))
mean = np.load(os.path.join(folder_path, "mean.npy"))
std = np.load(os.path.join(folder_path, "std.npy"))

with open(os.path.join(folder_path, "class_mapping.pkl"), 'rb') as f:
    class_to_idx = pickle.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

print(f"✅ Model loaded with {len(class_to_idx)} classes")
print(f"✅ First 10 classes: {list(class_to_idx.keys())[:10]}")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("\n🎥 Camera opened. Press 'q' to quit")
print("📝 Showing predictions on video feed")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    predicted_class = "No hand"
    confidence = 0.0
    color = (0, 0, 255)  # Default red for no hand
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks_norm = (landmarks - mean) / std
            
            # Predict
            pred = model.predict(landmarks_norm, verbose=0)[0]
            idx = np.argmax(pred)
            confidence = pred[idx]
            
            # Safeguard index
            if idx >= len(idx_to_class):
                print(f"⚠️ Warning: Predicted index {idx} out of range")
                idx = 0
                
            predicted_class = idx_to_class[idx]
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
    
    # Draw background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (350, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw predictions
    cv2.putText(frame, f"Sign: {predicted_class}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Show FPS (approximate)
    cv2.putText(frame, "Press 'q' to quit", (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('ISL Model Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✅ Test completed")