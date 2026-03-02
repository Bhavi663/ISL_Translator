# ISL Translator - Indian Sign Language Recognition System

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Model Information](#model-information)
- [Pages & Features](#pages--features)
- [Accessibility Features](#accessibility-features)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [File Structure](#file-structure)
- [Performance Metrics](#performance-metrics)
- [Technical Stack](#technical-stack)

## 🌟 Overview

The ISL Translator is a comprehensive web-based application designed to recognize and translate Indian Sign Language (ISL) gestures in real-time. Using deep learning and computer vision, it bridges communication gaps between the deaf/hard-of-hearing community and hearing individuals. The system supports 43 different classes including alphabets (A-Z), numbers (0-9), and common words/phrases.

### Key Features
- Real-time sign language recognition using webcam
- 43-class classification with 94.77% accuracy
- Multi-modal input (images, videos, text)
- Sentence building from recognized signs
- Text-to-Speech output
- Learning mode for practice
- Comprehensive analytics dashboard
- Full accessibility support

## 🏗 System Architecture

### Backend Components
- **Flask Server**: Handles HTTP requests, routing, and session management
- **TensorFlow Model**: Deep neural network for sign classification
- **MediaPipe**: Hand landmark detection and tracking
- **SQLite Database**: Stores translation history and user data
- **Authentication System**: User login/registration with session management

### Frontend Components
- **HTML5/CSS3**: Responsive UI with dark mode support
- **JavaScript**: Client-side interactivity and real-time updates
- **Chart.js/Highcharts**: Data visualization
- **Plotly**: 3D charts and radar plots
- **WebRTC**: Camera streaming

## 🤖 Model Information

### Neural Network Architecture
```python
Input Layer: 63 features (21 landmarks × 3 coordinates)
├── Dense(256) + ReLU + BatchNormalization + Dropout(0.3)
├── Dense(128) + ReLU + BatchNormalization + Dropout(0.3)
├── Dense(64) + ReLU + BatchNormalization + Dropout(0.3)
└── Output Layer: Dense(43) + Softmax
```

### Model Parameters
| Parameter | Value |
|-----------|-------|
| Total Parameters | 62,123 |
| Trainable Parameters | 61,227 |
| Non-trainable (BN) | 896 |
| Model Size | ~242 KB |
| Input Shape | (63,) |
| Output Classes | 43 |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Dataset Split | 80/20 |
| Training Samples | 3,600 |
| Test Samples | 900 |
| Validation Split | 15% (540 samples) |
| Batch Size | 32 |
| Epochs | 150 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Loss Function | Sparse CCE + Label Smoothing (0.1) |
| Dropout Rate | 0.3 |
| L2 Regularization | 1e-4 |
| Early Stopping | Patience 15 |
| LR Reduction | Factor 0.5, Patience 5, min_lr=1e-6 |

### Performance Metrics
| Metric | Value |
|--------|-------|
| Test Accuracy | 94.77% |
| Precision (Macro) | 94.2% |
| Recall (Macro) | 93.8% |
| F1-Score | 94.0% |
| Top-3 Accuracy | ~98% |
| Training Accuracy | 98.2% |
| Validation Accuracy | 96.5% |
| Inference Time | ~35ms per frame |
| FPS (Real-time) | 28.5 (CPU) |

## 📄 Pages & Features

### 1. **Live Translation Page** (`index.html`)
**Purpose**: Real-time sign language recognition via webcam

**Functions**:
- `startCamera()`: Initializes webcam and video stream
- `stopCamera()`: Stops camera and cleans up resources
- `startPredictionPolling()`: Polls server for predictions every 300ms
- `updatePredictionDisplay()`: Updates UI with current sign and confidence
- `speakText()`: Text-to-speech using browser's speech synthesis
- `toggleAutoSpeak()`: Enables/disables automatic speech for detected signs
- `checkSession()`: Verifies user authentication status

**Features**:
- Real-time hand detection and sign classification
- Confidence score display with color coding (green >80%, orange 50-80%, red <50%)
- Manual and auto-speak options
- TTS settings (voice, speed, volume)
- FPS counter
- Session management with auto-logout after 30 minutes

**UI Components**:
- Video wrapper with loading/error states
- Prediction panel with current sign and confidence
- TTS controls and settings
- Camera start/stop buttons
- Info panel with usage instructions
- Quick stats (classes, status, FPS)

### 2. **History Page** (`history.html`)
**Purpose**: Track and analyze past translations

**Functions**:
- `loadTranslations()`: Fetches user's translation history
- `updateStats()`: Calculates and displays statistics
- `updateMostCommonChart()`: Bar chart of most detected signs
- `updateAccuracyChart()`: Line chart of accuracy trend over 7 days
- `updateTable()`: Populates translations table
- `clearHistory()`: Deletes all translation history
- `exportData()`: Exports data as CSV or PDF
- `speakTranslation()`: TTS for individual signs in history

**Features**:
- 24-hour translation history view
- Statistics cards (total translations, most detected sign, average confidence)
- Interactive charts (most common signs, accuracy trend)
- Sortable/filterable table with confidence badges
- Export functionality (CSV, PDF)
- Real-time updates every 3 seconds
- Clear history option with confirmation

**Significance**: Enables users to track progress, identify frequently used signs, and monitor confidence trends over time.

### 3. **Performance Dashboard** (`performance.html`)
**Purpose**: Monitor model performance metrics and analytics

**Functions**:
- `initHighcharts()`: Initializes all charts with 3D visualization
- `updateMetricsTable()`: Populates class-wise metrics table
- `fetchLiveMetrics()`: Updates real-time KPI values
- `exportMetrics()`: Exports performance report as JSON
- `resetMetrics()`: Resets performance counters
- `speakMetrics()`: TTS for reading metrics aloud

**Charts**:
- Training History (accuracy/loss over 150 epochs)
- Accuracy vs Sample Size (scatter plot)
- Per-Class Performance (3D column chart)
- Class Distribution (3D pie chart)
- YOLO vs CNN Radar (comparison)
- Accuracy Distribution (histogram)

**KPIs**:
- Test Accuracy (94.77%)
- Precision (94.2%)
- Recall (93.8%)
- F1-Score (94.0%)
- Training Accuracy (98.2%)
- Validation Accuracy (96.5%)

**Significance**: Provides transparency into model performance, helps identify classes with lower accuracy, and enables data-driven improvements.

### 4. **Learning Mode** (`learn.html`)
**Purpose**: Interactive practice for learning ISL signs

**Functions**:
- `initAlphabetGrid()`: Creates alphabet reference grid
- `initNumbersGrid()`: Creates numbers reference grid
- `startPracticeCamera()`: Initializes camera for practice mode
- `startPredictionPolling()`: Monitors user's signs during practice
- `newChallenge()`: Generates random sign challenge
- `updateScore()`: Tracks correct/incorrect attempts
- `speakChallenge()`: TTS for challenge instructions

**Features**:
- Alphabet chart (A-Z) with visual references
- Numbers chart (0-9) with visual references
- Practice mode with timed challenges
- Real-time feedback on sign correctness
- Score tracking (correct, attempts, accuracy)
- Streak counter
- TTS for challenge instructions and progress

**Significance**: Provides an educational platform for learning ISL, making sign language accessible to everyone.

### 5. **Model Info Page** (`about-model.html`)
**Purpose**: Detailed technical documentation of the model

**Sections**:
- Hero section with model badges
- Key statistics (samples, classes, accuracy, FPS)
- Architecture diagram with 3D visualization
- Neural network layer details
- Dataset information
- Preprocessing pipeline
- Performance metrics table
- Limitations & future work
- Technology stack

**Features**:
- Interactive architecture diagrams
- Clickable images that open in new tab
- Dark mode support
- Detailed layer parameters
- Class-wise performance expectations

**Significance**: Provides transparency about model architecture, training process, and limitations for developers and researchers.

### 6. **Sentence Builder** (`sentence-builder.html`)
**Purpose**: Build complete sentences from recognized signs

**Functions**:
- `addToSentence()`: Adds current sign to sentence
- `removeLastWord()`: Removes last word from sentence
- `clearSentence()`: Clears current sentence
- `saveSentence()`: Saves sentence to history
- `speakFullSentence()`: TTS for complete sentence
- `fetchSuggestions()`: Provides word suggestions based on last sign
- `addPhrase()`: Adds predefined phrase to sentence

**Features**:
- Real-time sign detection
- Sentence display with word chips
- Context-aware suggestions
- Saved sentences history
- Quick phrases panel
- Common ISL phrases library
- TTS for individual signs and full sentences

**Significance**: Enables continuous communication by combining individual signs into meaningful sentences.

### 7. **Multi-Modal Page** (`multi-modal.html`)
**Purpose**: Process images, videos, and text input

**Functions**:
- `processImage()`: Upload and analyze image files
- `processVideo()`: Upload and analyze video files
- `processText()`: Text-to-speech and sign visualization
- `displaySignImage()`: Shows corresponding sign image for text
- `switchTab()`: Toggles between input modes

**Features**:
- Image upload with preview
- Video upload with playback
- Text input with sign image preview
- TTS for text input
- Confidence display for detected signs
- Prediction list for multiple detections

**Significance**: Provides flexibility for users who may not have webcam access or want to analyze pre-recorded content.

## ♿ Accessibility Features

### Keyboard Navigation
- **Tab Navigation**: All interactive elements are focusable via Tab key
- **Skip to Main Content**: Hidden link at page start for screen reader users
- **Focus Indicators**: Clear visual focus states for all buttons and links
- **Keyboard Shortcuts**: 
  - Enter/Space: Activate buttons
  - Escape: Close modals/dialogs
  - Arrow keys: Navigate charts and tables

### Screen Reader Compatibility
- **ARIA Labels**: All icons and buttons have descriptive ARIA labels
- **Semantic HTML**: Proper use of headings, lists, and landmarks
- **Alt Text**: All images have descriptive alt text
- **Live Regions**: Dynamic content updates announced to screen readers
- **Form Labels**: All form inputs have associated labels

### Color Blind Friendly
- **High Contrast**: Minimum 4.5:1 contrast ratio for all text
- **Color Independence**: Information not conveyed by color alone
- **Pattern Alternatives**: Charts use patterns in addition to colors
- **Dark Mode**: High-contrast dark theme option
- **Color Palette**: 
  - Primary: #ED985F (accessible for protanopia/deuteranopia)
  - Success: #10b981 (distinct from warning colors)
  - Warning: #ED985F
  - Error: #ef4444

### Text-to-Speech (TTS)
- **Browser Native**: Uses Web Speech API for offline TTS
- **Voice Selection**: Multiple voice options (female/male)
- **Rate Control**: Adjustable speech speed (0.5x to 2.0x)
- **Volume Control**: Adjustable output volume
- **Auto-Speak**: Optional automatic speech for detected signs
- **Fallback**: Server-side TTS when browser TTS unavailable

### Visual Accessibility
- **Font Scaling**: Responsive design supports browser font scaling
- **Readable Typography**: Inter font family with adequate line height
- **Whitespace**: Generous spacing between elements
- **Focus Indicators**: High-visibility focus rings
- **Animation Control**: Reduced motion option for vestibular disorders

### Cognitive Accessibility
- **Clear Instructions**: Step-by-step guides on each page
- **Consistent Navigation**: Same navigation bar across all pages
- **Error Prevention**: Confirmation dialogs for destructive actions
- **Feedback**: Clear success/error messages
- **Progress Tracking**: Visual progress indicators for timed activities

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Webcam (for live features)
- Modern web browser (Chrome, Firefox, Edge)

### Step-by-Step Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/isl-translator.git
cd isl-translator
```

2. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Required Libraries**
```bash
pip install flask flask-sqlalchemy tensorflow mediapipe opencv-python numpy pandas scikit-learn pyttsx3 pillow
```

5. **Prepare Model Files**
```bash
# Create models directory
mkdir -p web_app/models

# Place your trained model in the models folder
# Expected structure:
# web_app/models/trained_YYYY-MM-DD_HH-MM-SS/
#   ├── isl_model.keras
#   ├── mean.npy
#   ├── std.npy
#   └── class_mapping.pkl
```

6. **Prepare Static Files**
```bash
# Create signs directory for reference images
mkdir -p web_app/static/signs

# Add sign images (A.jpg, B.jpg, ... 0.jpg, 1.jpg, ... Hello.jpg, etc.)
# Images should be in web_app/static/signs/
```

7. **Initialize Database**
```python
# Run Python interpreter and execute:
from app import app, db
with app.app_context():
    db.create_all()
```

8. **Run the Application**
```bash
# Navigate to web_app directory
cd web_app

# Run Flask app
python app.py
```

9. **Access the Application**
- Open browser and navigate to: `http://localhost:5000`
- Register a new account or use demo credentials
- Allow camera permissions when prompted

### Docker Setup (Alternative)

```bash
# Build Docker image
docker build -t isl-translator .

# Run container
docker run -p 5000:5000 isl-translator
```

### Environment Variables

Create a `.env` file in the root directory:

```env
SECRET_KEY=your_secret_key_here
DATABASE_URL=sqlite:///translations.db
SESSION_TIMEOUT=30
DEBUG=False
```

## 📁 File Structure

```
ISL_Translator/
├── web_app/
│   ├── __init__.py
│   ├── app.py                 # Main Flask application
│   ├── auth_system.py         # User authentication
│   ├── multi_modal_handler.py # Multi-modal processing
│   ├── models/                 # Trained model files
│   │   └── trained_*/         
│   │       ├── isl_model.keras
│   │       ├── mean.npy
│   │       ├── std.npy
│   │       └── class_mapping.pkl
│   ├── static/
│   │   ├── signs/              # Reference sign images
│   │   │   ├── A.jpg
│   │   │   ├── B.jpg
│   │   │   ├── ...
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   ├── Hello.jpg
│   │   │   └── Namaste.jpg
│   │   ├── style.css           # Global styles
│   │   ├── accessibility.css    # Accessibility styles
│   │   └── accessibility.js     # Accessibility features
│   ├── templates/
│   │   ├── index.html           # Live translation
│   │   ├── history.html         # Translation history
│   │   ├── performance.html      # Performance dashboard
│   │   ├── learn.html           # Learning mode
│   │   ├── about-model.html      # Model information
│   │   ├── sentence-builder.html # Sentence builder
│   │   ├── multi-modal.html      # Multi-modal input
│   │   ├── login.html            # Login page
│   │   ├── register.html         # Registration page
│   │   └── accessibility_panel.html # Accessibility panel
│   └── translations.db           # SQLite database
├── requirements.txt
├── README.md
└── .env
```

## 📊 Performance Metrics

### Class-wise Performance (Sample)
| Class | Precision | Recall | F1-Score | Support | Accuracy | Confidence |
|-------|-----------|--------|----------|---------|----------|------------|
| 0 | 95% | 94% | 94.5% | 85 | 96% | 92% |
| A | 96% | 95% | 95.5% | 92 | 97% | 93% |
| Hello | 97% | 96% | 96.5% | 107 | 98% | 94% |
| Namaste | 98% | 97% | 97.5% | 111 | 99% | 95% |

### Category Distribution
| Category | Classes | Samples |
|----------|---------|---------|
| Digits (0-9) | 10 | ~900 |
| Alphabets (A-Z) | 26 | ~2,340 |
| Words | 7 | ~860 |
| **Total** | **43** | **~4,100** |

## 🛠 Technical Stack

### Backend
- **Framework**: Flask 2.3.2
- **Database**: SQLAlchemy 2.0.19 + SQLite
- **ML Framework**: TensorFlow 2.13.0
- **Computer Vision**: OpenCV 4.8.0, MediaPipe 0.10.3
- **Authentication**: Custom auth system with session management
- **TTS**: pyttsx3 2.90

### Frontend
- **HTML5/CSS3**: Custom responsive design
- **JavaScript**: ES6+
- **Charts**: Chart.js 4.4.0, Highcharts 11.1.0, Plotly 2.27.1
- **Icons**: Font Awesome 6.0.0
- **Fonts**: Google Inter
- **TTS**: Web Speech API

### Development Tools
- **Version Control**: Git
- **Package Manager**: pip
- **Environment**: Python virtualenv
- **Testing**: pytest (optional)

## 📝 Usage Guide

### Quick Start
1. Register/Login to the application
2. Allow camera permissions
3. Show a sign to the camera
4. View detected sign and confidence
5. Use "Speak Sign" to hear the sign
6. Enable "Auto" for automatic speech

### Building Sentences
1. Start camera and make signs
2. Click "Add to Sentence" to add each sign
3. Use suggestions for common word combinations
4. Save sentences for later use
5. Use "Speak Sentence" to hear the complete sentence

### Learning Mode
1. Switch to "Practice Mode"
2. Start camera
3. Follow the challenge prompts
4. Get real-time feedback on your signs
5. Track your progress with scores and streaks

### Multi-Modal Input
1. Switch to Image/Video/Text tab
2. Upload file or enter text
3. Click "Process"
4. View detection results and confidence
5. Use TTS to hear the translation

## 🔒 Security Features

- **Session Management**: 30-minute timeout with automatic logout
- **Password Security**: bcrypt hashing
- **SQL Injection Prevention**: SQLAlchemy ORM
- **XSS Protection**: Jinja2 autoescaping
- **CSRF Protection**: Secret key validation
- **HTTPS Ready**: Configurable for production

## 🌐 Browser Support

| Browser | Version | Support |
|---------|---------|---------|
| Chrome | 90+ | Full |
| Firefox | 88+ | Full |
| Edge | 90+ | Full |
| Safari | 14+ | Full (limited TTS) |

## 📈 Future Improvements

- [ ] Multi-hand tracking support
- [ ] Sentence-level translation (LSTM/RNN)
- [ ] Expand to 100+ ISL words/phrases
- [ ] Mobile app (React Native/Flutter)
- [ ] Real-time translation recording
- [ ] Community contribution for new signs
- [ ] Model quantization for mobile deployment
- [ ] Offline PWA support
- [ ] Multi-language UI (Hindi, Tamil, etc.)
- [ ] Voice-to-sign translation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Project Lead: [Your Name]
- ML Engineers: [Team Members]
- UI/UX Designers: [Team Members]
- Accessibility Consultants: [Team Members]

## 📞 Support

For issues and questions:
- GitHub Issues: [Repository URL]
- Email: support@isltranslator.com
- Documentation: [Wiki URL]

### APIs
## 🌐 API Endpoints Reference

The ISL Translator provides a comprehensive REST API for all functionality. Below is a detailed description of each endpoint.

### Authentication Endpoints

#### `POST /api/register`
**Description**: Register a new user account
**Access**: Public
**Request Body**:
```json
{
    "username": "john_doe",
    "password": "secure_password",
    "email": "john@example.com"  // optional
}
```
**Response (Success)**:
```json
{
    "success": true,
    "message": "User registered successfully",
    "username": "john_doe"
}
```
**Response (Error)**:
```json
{
    "success": false,
    "message": "Username already exists"
}
```

#### `POST /api/login`
**Description**: Authenticate user and create session
**Access**: Public
**Request Body**:
```json
{
    "username": "john_doe",
    "password": "secure_password"
}
```
**Response (Success)**:
```json
{
    "success": true,
    "message": "Login successful",
    "username": "john_doe",
    "session_duration": "30 minutes"
}
```
**Response (Error)**:
```json
{
    "success": false,
    "message": "Invalid credentials"
}
```

#### `POST /api/logout`
**Description**: End user session
**Access**: Protected
**Request Body**: None
**Response**:
```json
{
    "success": true,
    "message": "Logged out successfully"
}
```

#### `GET /api/session/status`
**Description**: Check if user is logged in
**Access**: Public
**Response (Logged In)**:
```json
{
    "logged_in": true,
    "user_id": "john_doe",
    "username": "john_doe",
    "login_time": "2026-02-28T10:30:00",
    "session_expiry": "2026-02-28T11:00:00"
}
```
**Response (Not Logged In)**:
```json
{
    "logged_in": false
}
```

### Camera & Video Streaming

#### `GET /start`
**Description**: Start camera and initialize video stream
**Access**: Protected
**Query Parameters**: None
**Response**:
```json
{
    "status": "started"
}
```

#### `GET /stop`
**Description**: Stop camera and clean up resources
**Access**: Protected
**Response**:
```json
{
    "status": "stopped"
}
```

#### `GET /video_feed`
**Description**: Streaming endpoint for live video feed (MJPEG)
**Access**: Protected
**Returns**: Multipart MJPEG stream with predictions overlay
```
Content-Type: multipart/x-mixed-replace; boundary=frame
```

#### `GET /current_prediction`
**Description**: Get the latest prediction from video stream
**Access**: Protected
**Response**:
```json
{
    "sign": "A",
    "confidence": 0.95
}
```

#### `GET /fps`
**Description**: Get current FPS of video processing
**Access**: Protected
**Response**:
```json
{
    "fps": 28.5
}
```

### Translation History

#### `GET /history/data`
**Description**: Get user's translation history (last 24 hours)
**Access**: Protected
**Response**:
```json
{
    "translations": [
        {
            "id": 1,
            "user_id": "john_doe",
            "sign": "A",
            "confidence": 95,
            "timestamp": "2026-02-28 10:30:00"
        },
        {
            "id": 2,
            "user_id": "john_doe",
            "sign": "Hello",
            "confidence": 92,
            "timestamp": "2026-02-28 10:31:00"
        }
    ],
    "total": 2,
    "most_common": [
        {"sign": "A", "count": 5},
        {"sign": "Hello", "count": 3}
    ],
    "daily_accuracy": [
        {"date": "Feb 22", "avg": 0.94},
        {"date": "Feb 23", "avg": 0.93}
    ]
}
```

#### `POST /api/save_translation`
**Description**: Save a single translation to history
**Access**: Protected
**Request Body**:
```json
{
    "sign": "A",
    "confidence": 95
}
```
**Response**:
```json
{
    "status": "success",
    "translation": {
        "sign": "A",
        "confidence": 95,
        "timestamp": "2026-02-28T10:32:00"
    }
}
```

#### `POST /api/clear_history`
**Description**: Clear all translation history for current user
**Access**: Protected
**Response**:
```json
{
    "status": "success"
}
```

#### `GET /api/export/<format>`
**Description**: Export history as CSV or JSON
**Access**: Protected
**Path Parameters**: `format` - "csv" or "json"
**Response**: File download

### Text-to-Speech (TTS)

#### `GET /api/tts-status`
**Description**: Check TTS engine status
**Access**: Public
**Response**:
```json
{
    "initialized": true,
    "rate": 150,
    "volume": 0.9
}
```

#### `POST /api/speak`
**Description**: Convert text to speech
**Access**: Protected
**Request Body**:
```json
{
    "text": "Hello, this is a test"
}
```
**Response**:
```json
{
    "status": "success",
    "message": "Speaking: Hello, this is a test"
}
```

#### `POST /api/tts-settings`
**Description**: Update TTS engine settings
**Access**: Protected
**Request Body**:
```json
{
    "voice": "female",      // or "male"
    "rate": 150,            // 50-200
    "volume": 0.9           // 0.0-1.0
}
```
**Response**:
```json
{
    "status": "success",
    "message": "TTS settings updated"
}
```

### Model & Performance

#### `GET /api/model_info`
**Description**: Get information about the current model
**Access**: Public
**Response**:
```json
{
    "name": "trained_2026-02-25_14-30-00",
    "num_classes": 43,
    "classes": ["0", "1", "2", ..., "Z"],
    "folder": "trained_2026-02-25_14-30-00"
}
```

#### `GET /api/performance_metrics`
**Description**: Get real-time performance metrics
**Access**: Public
**Response**:
```json
{
    "accuracy": 94.77,
    "precision": 94.2,
    "recall": 93.8,
    "f1": 94.0,
    "train_accuracy": 98.2,
    "fps": 28.5,
    "uptime": "02:34:56",
    "total_frames": 15420,
    "avg_confidence": 91.5
}
```

#### `GET /api/class_metrics`
**Description**: Get per-class performance metrics
**Access**: Public
**Response**:
```json
[
    {
        "class": "A",
        "accuracy": 97,
        "avg_confidence": 93,
        "total_samples": 92,
        "correct": 89
    },
    {
        "class": "B",
        "accuracy": 93,
        "avg_confidence": 89,
        "total_samples": 88,
        "correct": 82
    }
]
```

#### `GET /api/queue_status`
**Description**: Get database queue status
**Access**: Public
**Response**:
```json
{
    "size": 5,
    "avg_wait_time": 50,
    "processing_rate": 15.5,
    "max_size": 50
}
```

#### `POST /api/reset_metrics`
**Description**: Reset all performance counters
**Access**: Public
**Response**:
```json
{
    "status": "success",
    "message": "Metrics reset successfully"
}
```

#### `GET /api/models/list`
**Description**: List all available trained models
**Access**: Public
**Response**:
```json
[
    {
        "name": "trained_2026-02-25_14-30-00",
        "path": "/path/to/model",
        "timestamp": 1740576600
    },
    {
        "name": "trained_2026-02-24_10-15-00",
        "path": "/path/to/model",
        "timestamp": 1740492900
    }
]
```

### Sentence Builder

#### `GET /api/sentence/add/<path:sign>`
**Description**: Add a sign to current sentence
**Access**: Protected
**Path Parameters**: `sign` - The sign to add
**Response**:
```json
{
    "status": "success",
    "word": "A",
    "current_sentence": "A",
    "words": [
        {"original": "A", "mapped": "A"}
    ]
}
```

#### `GET /api/sentence/remove_last`
**Description**: Remove last word from sentence
**Access**: Protected
**Response**:
```json
{
    "status": "success",
    "removed": {"original": "A", "mapped": "A"},
    "current_sentence": "",
    "words": []
}
```

#### `GET /api/sentence/clear`
**Description**: Clear current sentence
**Access**: Protected
**Response**:
```json
{
    "status": "success",
    "message": "Sentence cleared"
}
```

#### `GET /api/sentence/save`
**Description**: Save current sentence to history
**Access**: Protected
**Response**:
```json
{
    "status": "success",
    "history": [
        {
            "sentence": "Hello how are you",
            "words": [...],
            "timestamp": "2026-02-28T10:35:00"
        }
    ]
}
```

#### `GET /api/sentence/history`
**Description**: Get saved sentences history
**Access**: Protected
**Response**:
```json
[
    {
        "sentence": "Hello how are you",
        "words": [...],
        "timestamp": "2026-02-28T10:35:00"
    },
    {
        "sentence": "My name is John",
        "words": [...],
        "timestamp": "2026-02-28T10:36:00"
    }
]
```

#### `GET /api/sentence/current`
**Description**: Get current sentence state
**Access**: Protected
**Response**:
```json
{
    "current_sentence": "Hello world",
    "words": [
        {"original": "Hello", "mapped": "Hello"},
        {"original": "world", "mapped": "world"}
    ]
}
```

#### `GET /api/sentence/suggestions/<path:sign>`
**Description**: Get word suggestions based on last sign
**Access**: Protected
**Path Parameters**: `sign` - Last detected sign
**Response**:
```json
{
    "suggestions": ["am", "want", "like", "need"]
}
```

### Multi-Modal Processing

#### `POST /api/process`
**Description**: Process image or video upload
**Access**: Protected
**Content-Type**: `multipart/form-data`
**Form Data**:
- `file`: Image/video file
- `type`: "image" or "video"

**Response (Image Success)**:
```json
{
    "success": true,
    "predictions": [
        {
            "class": "A",
            "confidence": 98.5,
            "hand": "Right"
        }
    ],
    "num_hands": 1
}
```

**Response (Video Success)**:
```json
{
    "success": true,
    "sentence": "Hello how are you",
    "signs": [
        {
            "sign": "Hello",
            "duration": 15,
            "confidence": 0.94
        },
        {
            "sign": "how",
            "duration": 12,
            "confidence": 0.92
        }
    ],
    "total_signs": 3,
    "duration_seconds": 5.2
}
```

**Response (Error)**:
```json
{
    "error": "No hand detected",
    "predictions": []
}
```

#### `GET /api/process/test`
**Description**: Test endpoint for multi-modal processing
**Access**: Protected
**Response**:
```json
{
    "status": "success",
    "message": "Multi-modal API is working",
    "endpoints": {
        "image": "POST /api/process with file and type=image",
        "video": "POST /api/process with file and type=video"
    }
}
```

### Attention Visualization

#### `POST /api/attention/toggle`
**Description**: Enable/disable attention overlay
**Access**: Protected
**Request Body**:
```json
{
    "enabled": true
}
```
**Response**:
```json
{
    "status": "success",
    "enabled": true
}
```

#### `GET /api/attention/status`
**Description**: Get current attention mode status
**Access**: Protected
**Response**:
```json
{
    "enabled": false
}
```

### Accessibility Settings

#### `POST /api/accessibility/save`
**Description**: Save accessibility preferences
**Access**: Public
**Request Body**:
```json
{
    "dark_mode": true,
    "large_text": false,
    "high_contrast": false,
    "reduced_motion": false
}
```
**Response**:
```json
{
    "status": "success",
    "settings": {
        "dark_mode": true,
        "large_text": false,
        "high_contrast": false,
        "reduced_motion": false
    }
}
```

#### `GET /api/accessibility/load`
**Description**: Load accessibility preferences
**Access**: Public
**Response**:
```json
{
    "dark_mode": true,
    "large_text": false,
    "high_contrast": false,
    "reduced_motion": false
}
```

#### `POST /api/accessibility/reset`
**Description**: Reset accessibility settings to default
**Access**: Public
**Response**:
```json
{
    "status": "success"
}
```

## 🔐 Authentication & Headers

### Protected Routes
All protected routes require an active session. Session is maintained via cookies:
```
Cookie: session=your_session_id
```

### Rate Limiting
- Public endpoints: 60 requests per minute
- Protected endpoints: 120 requests per minute
- Video streaming: Unlimited (optimized for continuous feed)

### Error Codes
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 401 | Unauthorized (not logged in) |
| 403 | Forbidden (insufficient permissions) |
| 404 | Not Found |
| 429 | Too Many Requests |
| 500 | Internal Server Error |

## 📡 WebSocket Events (Future Implementation)

Planned WebSocket endpoints for real-time bidirectional communication:
- `ws://server/predictions` - Live prediction stream
- `ws://server/performance` - Real-time metrics
- `ws://server/notifications` - System notifications

## 🧪 Testing the API

### Using cURL

```bash
# Login
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"john_doe","password":"secure_password"}'

# Get session status
curl http://localhost:5000/api/session/status

# Start camera
curl http://localhost:5000/start

# Get current prediction
curl http://localhost:5000/current_prediction

# Save translation
curl -X POST http://localhost:5000/api/save_translation \
  -H "Content-Type: application/json" \
  -d '{"sign":"A","confidence":95}'

# Speak text
curl -X POST http://localhost:5000/api/speak \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world"}'

# Logout
curl -X POST http://localhost:5000/api/logout
```

### Using Python Requests

```python
import requests

# Login
session = requests.Session()
response = session.post('http://localhost:5000/api/login', 
                       json={'username': 'john_doe', 'password': 'secure_password'})

# Get prediction
response = session.get('http://localhost:5000/current_prediction')
print(response.json())

# Upload image
files = {'file': open('sign.jpg', 'rb')}
data = {'type': 'image'}
response = session.post('http://localhost:5000/api/process', 
                       files=files, data=data)
print(response.json())
```

## 📊 API Performance

| Endpoint | Avg Response Time | Concurrent Users |
|----------|------------------|------------------|
| `/current_prediction` | 35ms | 100+ |
| `/api/speak` | 50ms | 50+ |
| `/api/process` (image) | 150ms | 20+ |
| `/api/process` (video) | 2-5s | 5+ |
| `/video_feed` | 33ms/frame | 10+ |

The API is optimized for real-time performance with:
- Connection pooling
- Response caching where appropriate
- Asynchronous database operations
- Worker threads for heavy processing


© 2026 ISL Translator - Breaking Communication Barriers