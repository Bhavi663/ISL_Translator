# multi_modal_handler.py
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import tensorflow as tf
from typing import Union, List, Dict
import io
import base64
import os

class MultiModalHandler:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        # Load model if exists
        models_dir = os.path.join(os.path.dirname(__file__), "web_app", "models")
        model_path = None
        
        if os.path.exists(models_dir):
            trained_folders = [f for f in os.listdir(models_dir) 
                              if f.startswith('trained_') and os.path.isdir(os.path.join(models_dir, f))]
            
            if trained_folders:
                latest_folder = sorted(trained_folders)[-1]
                model_path = os.path.join(models_dir, latest_folder, "isl_model.keras")
        
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            # Create fallback model
            inputs = tf.keras.Input(shape=(63,))
            x = tf.keras.layers.Dense(128, activation='relu')(inputs)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(43, activation='softmax')(x)
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Class mapping for 43 classes
        self.classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','Hello','I','India','J','K','L','M','N','Namaste','No','O','P','Please','Q','R','S','T','Thank_You','U','V','W','X','Y','Yes','Z']
        
    def process_image(self, image_data: Union[str, bytes, np.ndarray]) -> Dict:
        """Process single image input"""
        try:
            if isinstance(image_data, str):
                # Handle base64 encoded image
                if ',' in image_data:
                    image_data = base64.b64decode(image_data.split(',')[1])
                else:
                    image_data = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_data))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image = image_data
                
            # Extract hand landmarks
            results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_hand_landmarks:
                return {'error': 'No hand detected', 'predictions': []}
            
            predictions = []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract 63 features (21 landmarks * 3 coordinates)
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                
                # Make prediction
                pred = self.model.predict(np.array([features]), verbose=0)
                class_id = np.argmax(pred)
                confidence = float(pred[0][class_id])
                
                # Safeguard class_id
                if class_id >= len(self.classes):
                    class_id = 0
                
                predictions.append({
                    'class': self.classes[class_id],
                    'confidence': round(confidence * 100, 2),
                    'hand': 'Right' if i == 0 else 'Left'
                })
            
            return {
                'success': True,
                'predictions': predictions,
                'num_hands': len(predictions)
            }
        except Exception as e:
            return {'error': str(e), 'predictions': []}
    
    def process_video(self, video_data: Union[str, bytes], fps: int = 30) -> Dict:
        """Process video input - extracts keyframes and analyzes"""
        try:
            # Save temporary video file
            temp_path = 'temp_video.mp4'
            if isinstance(video_data, str) and ',' in video_data:
                # Handle base64 video
                video_data = base64.b64decode(video_data.split(',')[1])
                with open(temp_path, 'wb') as f:
                    f.write(video_data)
            elif isinstance(video_data, bytes):
                with open(temp_path, 'wb') as f:
                    f.write(video_data)
            else:
                return {'error': 'Unsupported video format'}
            
            # Open video capture
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract keyframes (every 30 frames)
            keyframe_indices = list(range(0, total_frames, 30))
            keyframes = []
            
            for idx in keyframe_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    keyframes.append(frame)
            
            cap.release()
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Process each keyframe
            sequence_predictions = []
            for frame in keyframes:
                result = self.process_image(frame)
                if 'predictions' in result and result['predictions']:
                    sequence_predictions.append(result['predictions'][0])
            
            # Analyze sequence for continuous signs
            return self.analyze_sequence(sequence_predictions)
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_sequence(self, predictions: List) -> Dict:
        """Analyze sequence of predictions for continuous signing"""
        if not predictions:
            return {'error': 'No valid predictions', 'sentence': ''}
        
        # Group consecutive similar predictions
        current_sign = None
        current_count = 0
        signs = []
        
        for pred in predictions:
            pred_class = pred['class']
            
            if pred_class == current_sign:
                current_count += 1
            else:
                if current_sign and current_count > 2:  # Minimum frames threshold
                    signs.append({
                        'sign': current_sign,
                        'duration': current_count,
                        'confidence': np.mean([p['confidence'] for p in predictions[-current_count:] if p['class'] == current_sign]) / 100
                    })
                current_sign = pred_class
                current_count = 1
        
        # Add last sign
        if current_sign and current_count > 2:
            signs.append({
                'sign': current_sign,
                'duration': current_count,
                'confidence': np.mean([p['confidence'] for p in predictions[-current_count:] if p['class'] == current_sign]) / 100
            })
        
        # Build sentence from detected signs (filter out digits for sentence)
        sentence_words = [s['sign'] for s in signs if s['sign'] not in ['0','1','2','3','4','5','6','7','8','9']]
        sentence = ' '.join(sentence_words)
        
        return {
            'success': True,
            'sentence': sentence,
            'signs': signs,
            'total_signs': len(signs),
            'duration_seconds': len(predictions) / 30  # Assuming 30 fps
        }
    
    def process_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Process multiple images in batch for efficiency"""
        results = []
        for image in images:
            result = self.process_image(image)
            if 'predictions' in result:
                results.extend(result['predictions'])
        return results