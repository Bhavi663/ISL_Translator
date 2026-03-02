import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os

# Load the trained model
model = tf.keras.models.load_model("models/isl_model.keras")
print("Model loaded successfully.")

# Load normalization parameters
mean = np.load("models/mean.npy")
std = np.load("models/std.npy")

# Load class mapping (to convert predicted index to class name)
with open("dataset/class_mapping.pkl", "rb") as f:
    class_to_idx = pickle.load(f)
# Reverse mapping: index -> class name
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set frame size for consistency
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Real-time translation started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Default text and color
    predicted_class = "No hand detected"
    confidence = 0.0
    color = (0, 0, 255)  # red

    if results.multi_hand_landmarks:
        # Take the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract landmarks (21 points * 3 coordinates = 63 values)
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Convert to numpy array and reshape
        landmarks_array = np.array(landmarks).reshape(1, -1)

        # Normalize using saved mean and std
        landmarks_normalized = (landmarks_array - mean) / std

        # Predict
        predictions = model.predict(landmarks_normalized, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_class = idx_to_class[predicted_idx]

        # Set color based on confidence (green if high, else orange)
        if confidence > 0.8:
            color = (0, 255, 0)  # green
        elif confidence > 0.5:
            color = (0, 165, 255)  # orange
        else:
            color = (0, 0, 255)  # red

        # Also display confidence
        confidence_text = f"{confidence:.2f}"

    else:
        confidence_text = ""

    # Display the predicted class and confidence on the frame
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {confidence_text}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show the frame
    cv2.imshow("Indian Sign Language Translator", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()