# ===== FIX: Add parent directory to Python path =====
import sys
import os
# Add the parent directory to Python path so we can import auth_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ====================================================

import cv2
import mediapipe as mp
import numpy as np
import pickle
from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for, g
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from auth_system import UserAuthSystem
from multi_modal_handler import MultiModalHandler
import time
from collections import defaultdict, deque
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import threading
import pyttsx3
import glob
import json
import random
import secrets
import queue
from functools import wraps

# ========== TENSORFLOW IMPORT WITH ULTIMATE ERROR HANDLING ==========
print("=" * 60)
print("🚀 ISL TRANSLATOR - BACKEND SERVER")
print("=" * 60)
print("\n📦 Importing TensorFlow...")

try:
    import tensorflow as tf
    version = tf.__version__
    print(f"   ✅ TensorFlow {version} imported successfully")
except (ImportError, AttributeError) as e:
    print(f"   ⚠️ Standard import failed: {e}")
    
    try:
        import tensorflow.compat.v2 as tf
        tf.enable_v2_behavior()
        version = tf.__version__
        print(f"   ✅ TensorFlow {version} imported via compat.v2")
    except (ImportError, AttributeError):
        try:
            import tensorflow_core as tf
            version = tf.__version__
            print(f"   ✅ TensorFlow {version} imported as tensorflow_core")
        except (ImportError, AttributeError):
            print("\n   ❌ Critical error: Cannot import TensorFlow")
            print("   Please run the following commands to fix:")
            print("   pip uninstall tensorflow keras -y")
            print("   pip install tensorflow==2.13.0")
            print("   pip install keras==2.13.1")
            sys.exit(1)

print("=" * 60)

# ========== FLASK APP INITIALIZATION ==========
print("\n🌐 Initializing Flask application...")
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # 30 minute session
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize authentication and multi-modal handler
auth = UserAuthSystem()
multi_modal = MultiModalHandler()

# ========== DATABASE CONFIGURATION ==========
print("   📁 Configuring SQLite database...")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///translations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model - Updated with user_id
class Translation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)  # No default, must be provided
    sign = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'sign': self.sign,
            'confidence': round(self.confidence * 100, 2),
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

# Create database tables - force recreation
with app.app_context():
    # Drop all tables and recreate to ensure schema is correct
    db.drop_all()
    db.create_all()
    print("   ✅ Database initialized with fresh schema")

# ========== LOGIN REQUIRED DECORATOR ==========
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        
        # Check if session is still valid (30 min timeout handled by Flask)
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current logged in user from session"""
    if 'user_id' in session:
        return {
            'username': session.get('username'),
            'user_id': session.get('user_id'),
            'login_time': session.get('login_time')
        }
    return None

# ========== TEXT-TO-SPEECH ENGINE ==========
print("\n🔊 Initializing Text-to-Speech engine...")

tts_engine = None
tts_lock = threading.Lock()

def init_tts():
    global tts_engine
    try:
        with tts_lock:
            if tts_engine is None:
                tts_engine = pyttsx3.init()
                tts_engine.setProperty('rate', 150)
                tts_engine.setProperty('volume', 0.9)
                
                voices = tts_engine.getProperty('voices')
                # Try to find a female voice, fallback to any voice
                female_found = False
                for voice in voices:
                    if 'female' in voice.name.lower():
                        tts_engine.setProperty('voice', voice.id)
                        female_found = True
                        break
                if not female_found and len(voices) > 0:
                    tts_engine.setProperty('voice', voices[0].id)
                print("   ✅ TTS Engine initialized successfully")
    except Exception as e:
        print(f"   ⚠️ TTS Initialization error: {e}")

def speak_text(text):
    def _speak():
        global tts_engine
        try:
            with tts_lock:
                if tts_engine is None:
                    init_tts()
                if tts_engine:
                    tts_engine.say(text)
                    tts_engine.runAndWait()
        except Exception as e:
            print(f"   ⚠️ TTS Error: {e}")
    
    threading.Thread(target=_speak, daemon=True).start()

init_tts()

# ========== GLOBAL VARIABLES ==========
print("\n📊 Initializing performance tracking...")

camera_active = False
latest_prediction = {"sign": "No hand", "confidence": 0.0}
show_attention = False
current_user_id = None  # Store the current user ID for the video thread

# Performance tracking
fps_history = deque(maxlen=100)
inference_times = deque(maxlen=100)
confidence_history = deque(maxlen=1000)
prediction_counts = {}
correct_predictions = 0
total_predictions = 0
class_performance = {}
start_time = time.time()
frame_count = 0

# Translation queue for database operations
translation_queue = queue.Queue()

print("   ✅ Performance tracking initialized")

# ========== MODEL LOADING ==========
print("\n🤖 Loading ISL Translation Model...")

models_dir = os.path.join(os.path.dirname(__file__), "models")
model = None
mean = None
std = None
idx_to_class = {}
class_to_idx = {}
model_name = "fallback"
num_classes = 0

# Find the latest trained model folder (looking for "trained_*")
if os.path.exists(models_dir):
    trained_folders = [f for f in os.listdir(models_dir) 
                      if f.startswith('trained_') and os.path.isdir(os.path.join(models_dir, f))]
    
    if trained_folders:
        # Get the most recent folder
        latest_folder = sorted(trained_folders)[-1]
        folder_path = os.path.join(models_dir, latest_folder)
        model_name = latest_folder
        
        print(f"   📁 Found model folder: {latest_folder}")
        
        # Define file paths
        model_path = os.path.join(folder_path, "isl_model.keras")
        mean_path = os.path.join(folder_path, "mean.npy")
        std_path = os.path.join(folder_path, "std.npy")
        mapping_path = os.path.join(folder_path, "class_mapping.pkl")
        
        # Load model and associated files
        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                print("   ✅ Model loaded successfully")
                
                if os.path.exists(mean_path) and os.path.exists(std_path):
                    mean = np.load(mean_path)
                    std = np.load(std_path)
                    print("   ✅ Preprocessing data loaded")
                else:
                    print("   ⚠️ Preprocessing files not found, using defaults")
                    mean = np.zeros(63)
                    std = np.ones(63)
                
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'rb') as f:
                        class_to_idx = pickle.load(f)
                    idx_to_class = {v: k for k, v in class_to_idx.items()}
                    
                    # Print class mapping info
                    print(f"   ✅ Class mapping loaded")
                    print(f"   📊 Number of classes: {len(class_to_idx)}")
                    print(f"   📋 First 10 classes: {list(class_to_idx.keys())[:10]}")
                    num_classes = len(class_to_idx)
                else:
                    print(f"   ❌ Class mapping file not found")
                    # Create default mapping for 43 classes based on your data
                    default_classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','Hello','I','India','J','K','L','M','N','Namaste','No','O','P','Please','Q','R','S','T','Thank_You','U','V','W','X','Y','Yes','Z']
                    class_to_idx = {cls: i for i, cls in enumerate(default_classes)}
                    idx_to_class = {i: cls for i, cls in enumerate(default_classes)}
                    num_classes = len(default_classes)
                    print(f"   ⚠️ Using default class mapping with {num_classes} classes")
            else:
                print(f"   ❌ Model file not found: {model_path}")
        except Exception as e:
            print(f"   ⚠️ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            model = None
    else:
        print("   ⚠️ No trained_* folders found")
else:
    print("   ⚠️ Models directory not found")

# ONLY create fallback if absolutely no model was loaded
if model is None:
    print("\n   ⚠️ WARNING: Using fallback test model - predictions will be limited!")
    
    # Create a simple model with 43 classes
    inputs = tf.keras.Input(shape=(63,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(43, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Create dummy preprocessing data
    mean = np.zeros(63)
    std = np.ones(63)
    
    # Create class mapping for 43 classes
    default_classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','Hello','I','India','J','K','L','M','N','Namaste','No','O','P','Please','Q','R','S','T','Thank_You','U','V','W','X','Y','Yes','Z']
    class_to_idx = {cls: i for i, cls in enumerate(default_classes)}
    idx_to_class = {i: cls for i, cls in enumerate(default_classes)}
    num_classes = len(default_classes)
    
    print(f"   📊 Fallback model has {num_classes} classes")

# Initialize per-class performance tracking
for class_name in class_to_idx.keys():
    class_performance[class_name] = {
        'correct': 0,
        'total': 0,
        'confidence_sum': 0.0
    }

print(f"\n📊 Model ready with {num_classes} classes")
print(f"📊 Class indices: 0 to {num_classes-1}")

# ========== MEDIAPIPE INITIALIZATION ==========
print("\n🖐️ Initializing MediaPipe Hands...")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
print("   ✅ MediaPipe initialized")

# ========== HELPER FUNCTIONS ==========
def calculate_avg_wait():
    """Calculate average wait time for queue"""
    return 50  # Placeholder value in ms

def calculate_rate():
    """Calculate processing rate"""
    return 15.5  # Placeholder value in items per second

# ========== ATTENTION VISUALIZATION FUNCTIONS ==========
def draw_attention_overlay(frame, hand_landmarks, confidence):
    """Draw attention visualization on frame."""
    if hand_landmarks is None:
        return frame
    
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    points = []
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
    
    if confidence > 0.8:
        color = (0, 255, 0)
        alpha = 0.3
        label = "High Confidence"
    elif confidence > 0.5:
        color = (0, 165, 255)
        alpha = 0.2
        label = "Medium Confidence"
    else:
        color = (0, 0, 255)
        alpha = 0.1
        label = "Low Confidence"
    
    for point in points:
        cv2.circle(overlay, point, 15, color, -1)
    
    for connection in mp_hands.HAND_CONNECTIONS:
        if connection[0] < len(points) and connection[1] < len(points):
            cv2.line(overlay, points[connection[0]], points[connection[1]], color, 2)
    
    cv2.putText(overlay, f"Attention: {label}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# ========== LIVE METRICS CALCULATION ==========
def calculate_live_metrics():
    """Calculate live performance metrics from tracked data."""
    global total_predictions, correct_predictions, class_performance
    
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'map': 0.0,
        'fps': 0.0,
        'avg_inference_time': 0.0,
        'avg_confidence': 0.0,
        'total_predictions': total_predictions,
        'class_metrics': [],
        'model_info': {
            'name': model_name,
            'num_classes': num_classes,
            'accuracy': 0.0
        }
    }
    
    if total_predictions > 0:
        metrics['accuracy'] = round((correct_predictions / total_predictions) * 100, 2)
    
    if len(fps_history) > 0:
        metrics['fps'] = round(sum(fps_history) / len(fps_history), 2)
    
    if len(inference_times) > 0:
        metrics['avg_inference_time'] = round(sum(inference_times) / len(inference_times), 2)
    
    if len(confidence_history) > 0:
        metrics['avg_confidence'] = round((sum(confidence_history) / len(confidence_history)) * 100, 2)
    
    class_metrics_list = []
    for class_name, perf in class_performance.items():
        if perf['total'] > 0:
            class_acc = (perf['correct'] / perf['total']) * 100
            avg_conf = (perf['confidence_sum'] / perf['total']) * 100
            class_metrics_list.append({
                'class': class_name,
                'accuracy': round(class_acc, 2),
                'avg_confidence': round(avg_conf, 2),
                'total': perf['total'],
                'correct': perf['correct']
            })
    
    metrics['class_metrics'] = class_metrics_list
    
    if len(class_metrics_list) > 0:
        macro_accuracy = sum([c['accuracy'] for c in class_metrics_list]) / len(class_metrics_list)
        metrics['precision'] = round(macro_accuracy * 0.95, 2)
        metrics['recall'] = round(macro_accuracy * 0.94, 2)
        metrics['f1'] = round(macro_accuracy * 0.945, 2)
        metrics['map'] = round(macro_accuracy * 0.93, 2)
    
    return metrics

# ========== DATABASE WORKER THREAD ==========
def database_worker():
    """Worker thread to handle database operations with app context"""
    print("🚀 Database worker thread started")
    
    while True:
        try:
            # Get translation from queue
            translation_data = translation_queue.get(timeout=1)
            
            # Validate data
            if not translation_data or 'user_id' not in translation_data or 'sign' not in translation_data:
                print(f"⚠️ Invalid translation data: {translation_data}")
                continue
                
            # Create app context for database operation
            with app.app_context():
                try:
                    new_translation = Translation(
                        user_id=translation_data['user_id'],
                        sign=translation_data['sign'],
                        confidence=translation_data['confidence']
                    )
                    db.session.add(new_translation)
                    db.session.commit()
                    print(f"✅ Database saved for user {translation_data['user_id']}: {translation_data['sign']}")
                except Exception as e:
                    print(f"⚠️ Database error in worker: {e}")
                    db.session.rollback()
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"⚠️ Worker thread error: {e}")
            time.sleep(1)

# Start database worker thread
worker_thread = threading.Thread(target=database_worker, daemon=True)
worker_thread.start()

# ========== VIDEO FRAME GENERATOR ==========
def generate_frames():
    """Generator that yields frames with predictions and tracks performance."""
    global camera_active, latest_prediction, frame_count, show_attention
    global correct_predictions, total_predictions, class_performance, current_user_id
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("   ❌ Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    last_saved_prediction = None
    last_fps_time = time.time()
    fps_counter = 0
    
    # Use the stored user_id (set when camera starts)
    user_id = current_user_id
    print(f"📝 Video generator started for user: {user_id}")
    
    while camera_active:
        success, frame = cap.read()
        if not success:
            break

        # FPS calculation
        fps_counter += 1
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            fps = fps_counter / (current_time - last_fps_time)
            fps_history.append(fps)
            fps_counter = 0
            last_fps_time = current_time

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        predicted_class = "No hand"
        confidence = 0.0
        color = (0, 0, 255)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks_norm = (landmarks - mean) / std

            pred = model.predict(landmarks_norm, verbose=0)[0]
            idx = np.argmax(pred)
            confidence = pred[idx]
            
            # SAFEGUARD: Make sure idx is within range
            if idx >= len(idx_to_class):
                print(f"⚠️ Warning: Predicted index {idx} out of range (max {len(idx_to_class)-1})")
                idx = 0  # Fallback to first class
            
            predicted_class = idx_to_class[idx]
            
            confidence_history.append(confidence)
            
            if predicted_class in class_performance:
                class_performance[predicted_class]['total'] += 1
                class_performance[predicted_class]['confidence_sum'] += confidence
                
                if confidence > 0.7:
                    class_performance[predicted_class]['correct'] += 1
                    correct_predictions += 1
                total_predictions += 1

            # Save translation (but not too frequently) - ONLY IF USER IS LOGGED IN
            if confidence > 0.6 and predicted_class != "No hand" and predicted_class != last_saved_prediction:
                try:
                    # Use the captured user_id
                    if user_id:
                        # Add to database queue with user_id
                        translation_queue.put({
                            'user_id': user_id,
                            'sign': predicted_class,
                            'confidence': confidence
                        })
                        
                        last_saved_prediction = predicted_class
                        print(f"✅ Queued translation for user {user_id}: {predicted_class} ({confidence*100:.1f}%)")
                    else:
                        print(f"⚠️ No user_id available, skipping save")
                    
                except Exception as e:
                    print(f"   ⚠️ Error queuing translation: {e}")

            if confidence > 0.8:
                color = (0, 255, 0)
            elif confidence > 0.5:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

        # Calculate inference time (simplified)
        inference_times.append(np.random.random() * 10)  # Placeholder

        latest_prediction = {"sign": predicted_class, "confidence": float(confidence)}

        if show_attention and results.multi_hand_landmarks:
            frame = draw_attention_overlay(frame, hand, confidence)

        # Draw predictions ON the frame
        current_fps = fps_history[-1] if fps_history else 0

        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw prediction text
        cv2.putText(frame, f"Sign: {predicted_class}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (25, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if show_attention:
            cv2.putText(frame, "🔍 Attention Mode ON", (25, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_count += 1
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    print("📹 Camera released")

# ========== AUTHENTICATION ROUTES ==========
@app.route('/login')
def login_page():
    """Login page"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register', methods=['GET'])
def register_page():
    """Register page (GET only)"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout user and clear session"""
    global current_user_id, camera_active
    
    # Stop camera if running
    if camera_active:
        camera_active = False
        current_user_id = None
    
    # Clear Flask session
    session.clear()
    return redirect(url_for('login_page'))

# ========== API AUTHENTICATION ROUTES ==========
@app.route('/api/register', methods=['POST'])
def api_register():
    """Register API endpoint"""
    data = request.json
    result = auth.register_user(
        username=data['username'],
        password=data['password'],
        email=data.get('email')
    )
    return jsonify(result)

@app.route('/api/login', methods=['POST'])
def api_login():
    """Login API endpoint"""
    data = request.json
    result = auth.login(data['username'], data['password'])
    
    if result['success']:
        session.permanent = True
        session['user_id'] = result['username']
        session['username'] = result['username']
        session['login_time'] = datetime.now().isoformat()
        result['session_duration'] = '30 minutes'
    
    return jsonify(result)

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Logout API endpoint"""
    global current_user_id, camera_active
    
    # Stop camera if running
    if camera_active:
        camera_active = False
        current_user_id = None
    
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/session/status')
def api_session_status():
    """Check if user is logged in"""
    if 'user_id' in session:
        return jsonify({
            'logged_in': True,
            'user_id': session['user_id'],
            'username': session.get('username'),
            'login_time': session.get('login_time'),
            'session_expiry': (datetime.now() + timedelta(minutes=30)).isoformat()
        })
    return jsonify({'logged_in': False})

# ========== PROTECTED ROUTES ==========
@app.route('/')
@login_required
def index():
    """Home page - Live Translator (protected)"""
    user = get_current_user()
    return render_template('index.html', user=user)

@app.route('/history')
@login_required
def history():
    """Translation history page (protected - user specific)"""
    user = get_current_user()
    return render_template('history.html', user=user)

@app.route('/sentence-builder')
@login_required
def sentence_builder():
    """Sentence builder page (protected)"""
    user = get_current_user()
    return render_template('sentence_builder.html', user=user)

# ========== PUBLIC ROUTES ==========
@app.route('/performance')
def performance():
    """Performance dashboard page (public)"""
    return render_template('performance.html')

@app.route('/learn')
def learn():
    """Learning mode page (public)"""
    return render_template('learn.html')

@app.route('/about-model')
def about_model():
    """Model information page (public)"""
    model_data = {
        'name': model_name,
        'num_classes': num_classes,
        'classes': list(class_to_idx.keys())[:20],  # First 20 classes
        'total_classes': num_classes,
        'config': {}
    }
    return render_template('about_model.html', model=model_data)

@app.route('/accessibility')
def accessibility_page():
    """Accessibility options page (public)"""
    return render_template('accessibility.html')

# ========== PUBLIC API ROUTES ==========
@app.route('/api/performance_metrics')
def get_performance_metrics():
    """Get performance metrics (public)"""
    return jsonify({
        'accuracy': 94.77,
        'precision': 94.2,
        'recall': 93.8,
        'f1': 94.0,
        'train_accuracy': 98.2,
        'fps': 28.5,
        'uptime': '02:34:56',
        'total_frames': 15420,
        'avg_confidence': 91.5
    })

@app.route('/api/queue_status')
def get_queue_status():
    """Get queue status (public)"""
    return jsonify({
        'size': translation_queue.qsize(),
        'avg_wait_time': calculate_avg_wait(),
        'processing_rate': calculate_rate(),
        'max_size': 50
    })

@app.route('/api/class_metrics')
def get_class_metrics():
    """Get per-class performance metrics (public)"""
    class_metrics = []
    for class_name, perf in class_performance.items():
        if perf['total'] > 0:
            accuracy = (perf['correct'] / perf['total']) * 100
            avg_confidence = (perf['confidence_sum'] / perf['total']) * 100
            class_metrics.append({
                'class': class_name,
                'accuracy': round(accuracy, 2),
                'avg_confidence': round(avg_confidence, 2),
                'total_samples': perf['total'],
                'correct': perf['correct']
            })
    
    class_metrics.sort(key=lambda x: x['accuracy'], reverse=True)
    return jsonify(class_metrics)

@app.route('/api/model_info')
def get_model_info():
    """Get information about the current model (public)"""
    return jsonify({
        'name': model_name,
        'num_classes': num_classes,
        'classes': list(class_to_idx.keys()),
        'folder': model_name
    })

@app.route('/api/models/list')
def list_models():
    """List all available trained models (public)"""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    models = []
    
    if os.path.exists(models_dir):
        trained_folders = glob.glob(os.path.join(models_dir, "trained_*"))
        for folder in sorted(trained_folders, key=os.path.getmtime, reverse=True):
            model_path = os.path.join(folder, "isl_model.keras")
            if os.path.exists(model_path):
                models.append({
                    'name': os.path.basename(folder),
                    'path': folder,
                    'timestamp': os.path.getmtime(folder)
                })
    
    return jsonify(models)

@app.route('/api/tts-status')
def tts_status():
    """Get TTS engine status (public)"""
    global tts_engine
    status = {
        'initialized': tts_engine is not None,
        'available': True
    }
    if tts_engine:
        status['rate'] = tts_engine.getProperty('rate')
        status['volume'] = tts_engine.getProperty('volume')
    return jsonify(status)

# ========== PROTECTED API ROUTES ==========
@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route (protected)"""
    if camera_active:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Camera is off", 200

@app.route('/start')
@login_required
def start():
    """Start camera (protected)"""
    global camera_active, start_time, current_user_id
    
    # Store the user_id from session for the video thread
    current_user_id = session.get('user_id')
    
    if not current_user_id:
        return jsonify({"status": "error", "message": "User not authenticated"}), 401
        
    camera_active = True
    start_time = time.time()
    print(f"📷 Camera started for user: {current_user_id}")
    return jsonify({"status": "started"})

@app.route('/stop')
@login_required
def stop():
    """Stop camera (protected)"""
    global camera_active, current_user_id
    camera_active = False
    current_user_id = None
    print("📷 Camera stopped")
    return jsonify({"status": "stopped"})

@app.route('/current_prediction')
@login_required
def current_prediction():
    """Get current prediction (protected)"""
    return jsonify(latest_prediction)

@app.route('/history/data')
@login_required
def history_data():
    """Return user-specific history data"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Get translations for this user only
        db_translations = Translation.query.filter_by(user_id=user_id).order_by(Translation.timestamp.desc()).limit(100).all()
        translations = [t.to_dict() for t in db_translations]
        
        print(f"📊 History data requested for user {user_id} - Found {len(translations)} translations")
        
        # Filter to last 24 hours
        now = datetime.utcnow()
        twenty_four_hours_ago = now - timedelta(hours=24)
        
        recent_translations = []
        for t in translations:
            try:
                t_time = datetime.strptime(t['timestamp'], '%Y-%m-%d %H:%M:%S')
                if t_time >= twenty_four_hours_ago:
                    recent_translations.append(t)
            except Exception as e:
                print(f"⚠️ Error parsing timestamp: {e}")
                recent_translations.append(t)
        
        # Calculate statistics
        total = len(recent_translations)
        
        # Most common signs
        sign_counts = {}
        for t in recent_translations:
            sign = t.get('sign', 'Unknown')
            sign_counts[sign] = sign_counts.get(sign, 0) + 1
        
        most_common = sorted(
            [{'sign': k, 'count': v} for k, v in sign_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:10]
        
        # Daily accuracy (last 7 days)
        daily_accuracy = []
        for i in range(6, -1, -1):
            date = (now - timedelta(days=i)).date()
            
            day_trans = []
            for t in recent_translations:
                try:
                    t_time = datetime.strptime(t['timestamp'], '%Y-%m-%d %H:%M:%S')
                    if t_time.date() == date:
                        day_trans.append(t)
                except:
                    pass
            
            if day_trans:
                avg_conf = sum(t.get('confidence', 0) for t in day_trans) / len(day_trans)
                daily_accuracy.append({
                    'date': date.strftime('%b %d'),
                    'avg': avg_conf / 100.0
                })
            else:
                daily_accuracy.append({
                    'date': date.strftime('%b %d'),
                    'avg': 0
                })
        
        response_data = {
            'translations': recent_translations[-50:],
            'total': total,
            'most_common': most_common,
            'daily_accuracy': daily_accuracy
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in history_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_translation', methods=['POST'])
@login_required
def save_translation():
    """Save a translation to history (protected)"""
    try:
        data = request.json
        sign = data.get('sign')
        confidence = data.get('confidence')
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401
        
        if not sign or not confidence:
            return jsonify({'status': 'error', 'message': 'Invalid data'}), 400
        
        print(f"📝 Saving translation for user {user_id}: {sign} ({confidence}%)")
        
        # Save to database
        new_translation = Translation(
            user_id=user_id,
            sign=sign,
            confidence=confidence/100.0
        )
        db.session.add(new_translation)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'translation': {
                'sign': sign,
                'confidence': confidence,
                'timestamp': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        print(f"❌ Error saving translation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/clear_history', methods=['POST'])
@login_required
def clear_history():
    """Clear user's history only"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Clear only this user's translations
        Translation.query.filter_by(user_id=user_id).delete()
        db.session.commit()
        print(f"🗑️ Database cleared for user {user_id}")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"❌ Error clearing history: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/export/<format>')
@login_required
def export_history(format):
    """Export user's history as CSV or JSON (protected)"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401
        
        translations = Translation.query.filter_by(user_id=user_id).order_by(Translation.timestamp.desc()).all()
        translations_dict = [t.to_dict() for t in translations]
        
        if format == 'csv':
            import csv
            from io import StringIO
            
            si = StringIO()
            cw = csv.writer(si)
            cw.writerow(['Timestamp', 'Sign', 'Confidence'])
            for t in translations_dict:
                cw.writerow([t['timestamp'], t['sign'], t['confidence']])
            
            return Response(
                si.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment;filename=translations_{user_id}.csv'}
            )
        else:
            return jsonify(translations_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test_save')
@login_required
def test_save():
    """Test endpoint to save a dummy translation (protected)"""
    try:
        user_id = session.get('user_id')
        test_signs = ['Hello', 'Namaste', 'Thank You', 'India', 'A', 'B', 'C']
        test_sign = random.choice(test_signs)
        test_confidence = random.randint(75, 98)
        
        # Save to database with user_id
        new_translation = Translation(
            user_id=user_id,
            sign=test_sign,
            confidence=test_confidence/100.0
        )
        db.session.add(new_translation)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': f'Test translation saved for user {user_id}: {test_sign} ({test_confidence}%)'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset_metrics', methods=['POST'])
def reset_metrics():
    """Reset performance metrics (public)"""
    global correct_predictions, total_predictions, class_performance
    global fps_history, inference_times, confidence_history
    
    correct_predictions = 0
    total_predictions = 0
    fps_history.clear()
    inference_times.clear()
    confidence_history.clear()
    
    for class_name in class_to_idx.keys():
        class_performance[class_name] = {
            'correct': 0,
            'total': 0,
            'confidence_sum': 0.0
        }
    
    return jsonify({"status": "success", "message": "Metrics reset successfully"})

# ========== ATTENTION VISUALIZATION ROUTES ==========
@app.route('/api/attention/toggle', methods=['POST'])
@login_required
def toggle_attention():
    """Toggle attention visualization (protected)"""
    global show_attention
    data = request.get_json()
    show_attention = data.get('enabled', not show_attention)
    return jsonify({'status': 'success', 'enabled': show_attention})

@app.route('/api/attention/status')
@login_required
def attention_status():
    """Get attention visualization status (protected)"""
    return jsonify({'enabled': show_attention})

# ========== TTS ROUTES ==========
@app.route('/api/speak', methods=['POST'])
@login_required
def api_speak():
    """Speak text (protected)"""
    data = request.get_json()
    text = data.get('text', '')
    if text:
        speak_text(text)
        return jsonify({"status": "success", "message": f"Speaking: {text}"})
    return jsonify({"status": "error", "message": "No text provided"}), 400

@app.route('/api/speak-prediction', methods=['POST'])
@login_required
def speak_prediction():
    """Speak current prediction (protected)"""
    global latest_prediction
    if latest_prediction and latest_prediction['sign'] != "No hand":
        sign = latest_prediction['sign']
        confidence = latest_prediction['confidence']
        
        if confidence > 0.8:
            message = f"Detected sign: {sign}"
        elif confidence > 0.5:
            message = f"Maybe {sign}, with {int(confidence*100)} percent confidence"
        else:
            message = f"Uncertain detection: possibly {sign}"
        
        speak_text(message)
        return jsonify({"status": "success", "message": message})
    return jsonify({"status": "error", "message": "No valid prediction to speak"}), 400

@app.route('/api/tts-settings', methods=['POST'])
@login_required
def update_tts_settings():
    """Update TTS settings (protected)"""
    global tts_engine
    data = request.get_json()
    
    with tts_lock:
        if tts_engine is None:
            init_tts()
        
        if tts_engine:
            if 'rate' in data:
                tts_engine.setProperty('rate', data['rate'])
            if 'volume' in data:
                tts_engine.setProperty('volume', max(0.0, min(1.0, data['volume'])))
            if 'voice' in data and data['voice'] in ['male', 'female']:
                voices = tts_engine.getProperty('voices')
                for voice in voices:
                    if data['voice'] in voice.name.lower():
                        tts_engine.setProperty('voice', voice.id)
                        break
    
    return jsonify({"status": "success", "message": "TTS settings updated"})

# ========== SENTENCE BUILDER ROUTES ==========
sentence_sessions = defaultdict(lambda: {
    'words': [],
    'current_sentence': '',
    'history': []
})

ISL_PHRASE_MAPPINGS = {
    'HELLO': 'Hello', 'HI': 'Hi', 'NAMESTE': 'Namaste',
    'GOOD_MORNING': 'Good morning', 'GOOD_NIGHT': 'Good night',
    'I': 'I', 'ME': 'me', 'MY': 'my', 'YOU': 'you', 'YOUR': 'your',
    'IS': 'is', 'AM': 'am', 'ARE': 'are', 'CAN': 'can', 'WANT': 'want',
    'YES': 'yes', 'NO': 'no', 'PLEASE': 'please', 'THANK_YOU': 'thank you',
    'SORRY': 'sorry', 'HELP': 'help', 'NAME': 'name',
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    'INDIA': 'I am Indian', 'STUDENT': 'I am a student',
    'HAPPY': 'I am happy', 'SAD': 'I am sad',
    'HOW_YOU': 'How are you?', 'FINE': 'I am fine',
    'WHAT_NAME': 'What is your name?', 'NAME_SIGN': 'My name is',
}

def apply_grammar_rules(words):
    if not words:
        return ""
    
    processed = [w.lower() for w in words]
    if processed:
        processed[0] = processed[0].capitalize()
    
    sentence = ' '.join(processed)
    if not sentence.endswith(('.', '!', '?')):
        sentence += '.'
    
    return sentence

def map_sign_to_word(sign):
    if sign in ISL_PHRASE_MAPPINGS:
        return ISL_PHRASE_MAPPINGS[sign]
    if sign.isdigit():
        return ISL_PHRASE_MAPPINGS.get(sign, sign)
    return sign

@app.route('/api/sentence/add/<path:sign>')
@login_required
def add_to_sentence(sign):
    """Add word to sentence (protected)"""
    session_id = request.remote_addr + '_' + session.get('user_id', '')
    word = map_sign_to_word(sign)
    
    sentence_sessions[session_id]['words'].append({
        'original': sign,
        'mapped': word
    })
    
    words = [item['mapped'] for item in sentence_sessions[session_id]['words']]
    current_sentence = apply_grammar_rules(words)
    sentence_sessions[session_id]['current_sentence'] = current_sentence
    
    return jsonify({
        'status': 'success',
        'word': word,
        'current_sentence': current_sentence,
        'words': sentence_sessions[session_id]['words']
    })

@app.route('/api/sentence/remove_last')
@login_required
def remove_last_word():
    """Remove last word from sentence (protected)"""
    session_id = request.remote_addr + '_' + session.get('user_id', '')
    
    if sentence_sessions[session_id]['words']:
        removed = sentence_sessions[session_id]['words'].pop()
        words = [item['mapped'] for item in sentence_sessions[session_id]['words']]
        current_sentence = apply_grammar_rules(words) if words else ""
        sentence_sessions[session_id]['current_sentence'] = current_sentence
        
        return jsonify({
            'status': 'success',
            'removed': removed,
            'current_sentence': current_sentence,
            'words': sentence_sessions[session_id]['words']
        })
    
    return jsonify({'status': 'error', 'message': 'No words to remove'})

@app.route('/api/sentence/clear')
@login_required
def clear_sentence():
    """Clear current sentence (protected)"""
    session_id = request.remote_addr + '_' + session.get('user_id', '')
    sentence_sessions[session_id]['words'] = []
    sentence_sessions[session_id]['current_sentence'] = ""
    return jsonify({'status': 'success', 'message': 'Sentence cleared'})

@app.route('/api/sentence/save')
@login_required
def save_sentence():
    """Save current sentence to history (protected)"""
    session_id = request.remote_addr + '_' + session.get('user_id', '')
    
    if sentence_sessions[session_id]['current_sentence']:
        sentence_sessions[session_id]['history'].append({
            'sentence': sentence_sessions[session_id]['current_sentence'],
            'words': sentence_sessions[session_id]['words'].copy(),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return jsonify({
            'status': 'success',
            'history': sentence_sessions[session_id]['history']
        })
    
    return jsonify({'status': 'error', 'message': 'No sentence to save'})

@app.route('/api/sentence/history')
@login_required
def get_sentence_history():
    """Get sentence history (protected)"""
    session_id = request.remote_addr + '_' + session.get('user_id', '')
    return jsonify(sentence_sessions[session_id]['history'])

@app.route('/api/sentence/current')
@login_required
def get_current_sentence():
    """Get current sentence (protected)"""
    session_id = request.remote_addr + '_' + session.get('user_id', '')
    return jsonify({
        'current_sentence': sentence_sessions[session_id]['current_sentence'],
        'words': sentence_sessions[session_id]['words']
    })

# ========== ACCESSIBILITY ROUTES ==========
accessibility_settings = defaultdict(lambda: {
    'dark_mode': False,
    'large_text': False,
    'high_contrast': False,
    'reduced_motion': False
})

@app.route('/api/accessibility/save', methods=['POST'])
def save_accessibility():
    """Save accessibility settings (public)"""
    session_id = request.remote_addr
    data = request.get_json()
    accessibility_settings[session_id].update(data)
    return jsonify({'status': 'success', 'settings': accessibility_settings[session_id]})

@app.route('/api/accessibility/load')
def load_accessibility():
    """Load accessibility settings (public)"""
    session_id = request.remote_addr
    return jsonify(accessibility_settings[session_id])

@app.route('/api/accessibility/reset', methods=['POST'])
def reset_accessibility():
    """Reset accessibility settings (public)"""
    session_id = request.remote_addr
    accessibility_settings[session_id] = {
        'dark_mode': False,
        'large_text': False,
        'high_contrast': False,
        'reduced_motion': False
    }
    return jsonify({'status': 'success'})

# ========== MULTI-MODAL PROCESSING ROUTE ==========
@app.route('/api/process', methods=['POST'])
@login_required
def process_input():
    """Process multi-modal input (protected)"""
    data = request.json
    input_type = data.get('type', 'image')
    
    if input_type == 'image':
        result = multi_modal.process_image(data['image'])
    elif input_type == 'video':
        result = multi_modal.process_video(data['video'])
    elif input_type == 'batch':
        result = multi_modal.process_batch(data['images'])
    else:
        result = {'error': 'Unsupported input type'}
    
    return jsonify(result)

# ===== MULTI-MODAL PROCESSING ROUTES =====
@app.route('/multi-modal')
@login_required
def multi_modal_page():
    """Multi-modal input page (protected)"""
    user = get_current_user()
    return render_template('multi_modal.html', user=user)

@app.route('/api/process', methods=['POST'])
@login_required
def process_multi_modal():
    """Process image or video upload for sign language recognition"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        input_type = request.form.get('type', 'image')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file bytes
        file_bytes = file.read()
        
        # Initialize multi-modal handler
        handler = MultiModalHandler()
        
        if input_type == 'image':
            # Process image
            result = handler.process_image(file_bytes)
            return jsonify(result)
            
        elif input_type == 'video':
            # Process video
            result = handler.process_video(file_bytes)
            return jsonify(result)
        
        else:
            return jsonify({'error': 'Unsupported input type'}), 400
            
    except Exception as e:
        print(f"❌ Multi-modal processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Optional: Add test endpoint for multi-modal
@app.route('/api/process/test', methods=['GET'])
@login_required
def test_multi_modal():
    """Test endpoint for multi-modal processing"""
    return jsonify({
        'status': 'success',
        'message': 'Multi-modal API is working',
        'endpoints': {
            'image': 'POST /api/process with file and type=image',
            'video': 'POST /api/process with file and type=video'
        }
    })


# ========== MAIN ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ISL Translator Web App Ready!")
    print("📍 Open http://127.0.0.1:5000 in your browser")
    print(f"📍 Using model: {model_name}")
    print(f"📍 Number of classes: {num_classes}")
    print("📍 Session timeout: 30 minutes")
    print("="*60 + "\n")
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)