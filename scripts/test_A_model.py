import numpy as np
import pickle
import os
import cv2
import mediapipe as mp
import tensorflow as tf

print("=" * 60)
print("🎥 TESTING LETTER A MODEL")
print("=" * 60)

# Find latest A model
models_dir = "models"
a_folders = [f for f in os.listdir(models_dir) if f.startswith("training_A_fixed")]
if not a_folders:
    print("❌ No A model found! Please train first.")
    exit()

latest_folder = sorted(a_folders)[-1]
model_path = os.path.join(models_dir, latest_folder, "isl_model.keras")
mean_path = os.path.join(models_dir, latest_folder, "mean.npy")
std_path = os.path.join(models_dir, latest_folder, "std.npy")

print(f"\n📁 Loading model from: {latest_folder}")

# Load model and preprocessing
model = tf.keras.models.load_model(model_path)
mean = np.load(mean_path)
std = np.load(std_path)

print("✅ Model loaded successfully")

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🎥 Webcam started. Make the 'A' sign to test!")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Draw landmarks if detected
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
            
            # Predict
            pred_prob = model.predict(landmarks_norm, verbose=0)[0][0]
            
            # Determine result
            if pred_prob > 0.5:
                label = "A"
                confidence = pred_prob
                color = (0, 255, 0)
            else:
                label = "NOT A"
                confidence = 1 - pred_prob
                color = (0, 0, 255)
            
            # Display
            cv2.putText(frame, f"Prediction: {label}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Test Letter A', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()