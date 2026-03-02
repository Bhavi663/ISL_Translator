import numpy as np
import pickle
import os
import cv2
import mediapipe as mp
import tensorflow as tf
from collections import deque
import glob

print("=" * 60)
print("🎥 PROPER TESTING FOR LETTER A")
print("=" * 60)

# Find latest balanced A model
models_dir = "models"
a_folders = [f for f in os.listdir(models_dir) if f.startswith("training_A_balanced")]
if not a_folders:
    print("❌ No balanced A model found! Please train first.")
    print("\nRun: python scripts\\train_A_balanced_fixed.py")
    exit()

latest_folder = sorted(a_folders)[-1]
model_path = os.path.join(models_dir, latest_folder, "isl_model.keras")
mean_path = os.path.join(models_dir, latest_folder, "mean.npy")
std_path = os.path.join(models_dir, latest_folder, "std.npy")
mapping_path = os.path.join(models_dir, latest_folder, "class_mapping.pkl")

print(f"\n📁 Loading model from: {latest_folder}")

# Check if files exist
if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    exit()

# Load model and preprocessing
try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
    
    mean = np.load(mean_path)
    std = np.load(std_path)
    print("✅ Preprocessing data loaded")
    
    with open(mapping_path, 'rb') as f:
        class_mapping = pickle.load(f)
    print(f"✅ Class mapping loaded: {class_mapping}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# For smoothing predictions
prediction_history = deque(maxlen=10)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🎥 Webcam started.")
print("\n📝 Instructions:")
print("   - Make the 'A' sign: closed fist with thumb extended")
print("   - Make other signs (B, C, peace sign, etc.) to test negative cases")
print("   - Press 'q' to quit")
print("-" * 60)

frame_count = 0
a_detections = 0
not_a_detections = 0

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
            try:
                pred_prob = model.predict(landmarks_norm, verbose=0)[0][0]
                
                # Add to history for smoothing
                prediction_history.append(pred_prob)
                
                # Use average of last few predictions
                avg_prob = np.mean(prediction_history)
                
                # Determine result with threshold
                if avg_prob > 0.7:  # Higher threshold for A
                    label = "A"
                    confidence = avg_prob
                    color = (0, 255, 0)
                    a_detections += 1
                elif avg_prob < 0.3:  # Lower threshold for NOT A
                    label = "NOT A"
                    confidence = 1 - avg_prob
                    color = (0, 0, 255)
                    not_a_detections += 1
                else:
                    label = "UNCERTAIN"
                    confidence = max(avg_prob, 1-avg_prob)
                    color = (0, 165, 255)
                
                # Display
                cv2.putText(frame, f"Prediction: {label}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show raw probability
                cv2.putText(frame, f"Raw A prob: {avg_prob:.2f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show counts
                cv2.putText(frame, f"A: {a_detections} | NOT A: {not_a_detections}", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            except Exception as e:
                cv2.putText(frame, f"Prediction error", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        prediction_history.clear()
    
    # Instructions
    cv2.putText(frame, "Make A sign (fist + thumb) | Any other sign = NOT A", 
                (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit", 
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow(f'Letter A Test - Using model: {latest_folder}', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("📊 TEST SUMMARY")
print("="*60)
print(f"   A detections: {a_detections}")
print(f"   NOT A detections: {not_a_detections}")
print(f"   Total frames with hand: {a_detections + not_a_detections}")
print("="*60)