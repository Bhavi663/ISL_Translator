#!/usr/bin/env python3
"""
Test ISL Model with Real Collected Data
--------------------------------------
This script tests your trained model on real collected sign language data.
"""

import os
import sys
import pickle
import numpy as np
import cv2
import mediapipe as mp
from collections import defaultdict
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} imported")
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")
    sys.exit(1)

class RealTimeTester:
    """Test model with real webcam data"""
    
    def __init__(self, model_folder=None):
        self.model = None
        self.mean = None
        self.std = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Statistics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.confidence_history = []
        
        self.load_model(model_folder)
    
    def load_model(self, model_folder=None):
        """Load trained model"""
        if model_folder is None:
            # Find latest model
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            trained_folders = [f for f in os.listdir(models_dir) 
                              if f.startswith('trained_')]
            if not trained_folders:
                print("❌ No trained models found")
                return False
            model_folder = os.path.join(models_dir, sorted(trained_folders)[-1])
        
        print(f"\n📁 Loading model from: {model_folder}")
        
        # Load model files
        model_path = os.path.join(model_folder, "isl_model.keras")
        mean_path = os.path.join(model_folder, "mean.npy")
        std_path = os.path.join(model_folder, "std.npy")
        mapping_path = os.path.join(model_folder, "class_mapping.pkl")
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
            
            with open(mapping_path, 'rb') as f:
                self.class_to_idx = pickle.load(f)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            
            print(f"✅ Model loaded: {len(self.class_to_idx)} classes")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks)
        return None
    
    def predict(self, landmarks):
        """Make prediction from landmarks"""
        if landmarks is None:
            return None, 0
        
        # Normalize
        landmarks_norm = (landmarks.reshape(1, -1) - self.mean) / self.std
        
        # Predict
        pred = self.model.predict(landmarks_norm, verbose=0)[0]
        idx = np.argmax(pred)
        confidence = pred[idx]
        
        return self.idx_to_class[idx], confidence
    
    def run_test(self, num_samples=100, save_results=True):
        """Run real-time test with webcam"""
        print("\n" + "="*60)
        print("🎥 REAL-TIME MODEL TESTING")
        print("="*60)
        print(f"Target samples: {num_samples}")
        print("Press SPACE to record a prediction, ESC to exit")
        print("-"*60)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n⏳ Testing in progress... Press ESC when done.\n")
        
        collected = []
        
        while self.total_predictions < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmarks = self.extract_landmarks(frame)
            
            if landmarks is not None:
                sign, confidence = self.predict(landmarks)
                
                # Display
                cv2.putText(frame, f"Sign: {sign}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Sample: {self.total_predictions}/{num_samples}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Record on spacebar
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    actual = input(f"\n❓ What sign did you make? (Predicted: {sign}) → ")
                    if actual:
                        collected.append({
                            'landmarks': landmarks.tolist(),
                            'predicted': sign,
                            'actual': actual.upper(),
                            'confidence': float(confidence)
                        })
                        
                        # Update stats
                        self.total_predictions += 1
                        self.class_stats[actual.upper()]['total'] += 1
                        if actual.upper() == sign and confidence > 0.7:
                            self.correct_predictions += 1
                            self.class_stats[actual.upper()]['correct'] += 1
                        
                        print(f"   ✅ Recorded #{self.total_predictions}")
                elif key == 27:  # ESC
                    break
            
            cv2.imshow('Model Testing - Press SPACE to record', frame)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Save results
        if save_results and collected:
            self.save_results(collected)
        
        return collected
    
    def calculate_metrics(self):
        """Calculate test metrics"""
        print("\n" + "="*60)
        print("📊 TEST RESULTS")
        print("="*60)
        print(f"Total samples: {self.total_predictions}")
        print(f"Correct predictions: {self.correct_predictions}")
        
        if self.total_predictions > 0:
            accuracy = (self.correct_predictions / self.total_predictions) * 100
            print(f"Accuracy: {accuracy:.2f}%")
        
        print("\n📈 Per-Class Performance:")
        print("-"*40)
        print(f"{'Class':<10} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
        print("-"*40)
        
        for class_name, stats in sorted(self.class_stats.items()):
            if stats['total'] > 0:
                acc = (stats['correct'] / stats['total']) * 100
                print(f"{class_name:<10} {stats['total']:<8} {stats['correct']:<8} {acc:>6.1f}%")
    
    def save_results(self, collected):
        """Save test results to file"""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'total_samples': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': (self.correct_predictions/self.total_predictions*100) if self.total_predictions > 0 else 0,
            'class_stats': dict(self.class_stats),
            'samples': collected
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {filename}")

def main():
    print("\n" + "="*60)
    print("🎯 REAL ISL MODEL TESTER")
    print("="*60)
    
    tester = RealTimeTester()
    
    if not tester.model:
        print("❌ Failed to load model")
        return
    
    # Ask for number of samples
    try:
        num_samples = int(input("\nHow many samples to collect? (default: 50): ") or "50")
    except:
        num_samples = 50
    
    # Run test
    tester.run_test(num_samples)

if __name__ == "__main__":
    main()