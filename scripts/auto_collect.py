import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import time

class AutoDataCollector:
    def __init__(self):
        self.base_dir = "dataset"
        self.raw_dir = os.path.join(self.base_dir, "raw_images")
        self.processed_dir = os.path.join(self.base_dir, "processed_landmarks")
        
        # Create directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Predefined classes
        self.classes = {
            '1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E', '6': 'F', '7': 'G', '8': 'H', '9': 'I', '10': 'J',
            '11': 'K', '12': 'L', '13': 'M', '14': 'N', '15': 'O', '16': 'P', '17': 'Q', '18': 'R', '19': 'S', '20': 'T',
            '21': 'U', '22': 'V', '23': 'W', '24': 'X', '25': 'Y', '26': 'Z',
            '27': '0', '28': '1', '29': '2', '30': '3', '31': '4', '32': '5', '33': '6', '34': '7', '35': '8', '36': '9',
            '37': 'Hello', '38': 'Namaste', '39': 'Thank_You', '40': 'Please', '41': 'Sorry',
            '42': 'Yes', '43': 'No', '44': 'Help', '45': 'India', '46': 'Student',
            '47': 'Teacher', '48': 'Father', '49': 'Mother', '50': 'Brother', '51': 'Sister'
        }
        
    def select_class(self):
        """Interactive class selection"""
        print("\n" + "="*60)
        print("📝 SELECT SIGN TO COLLECT")
        print("="*60)
        
        # Show available classes
        print("\nAvailable Classes:")
        print("-" * 40)
        
        # Show in groups
        print("\n🔤 ALPHABETS (A-Z):")
        for i in range(1, 27):
            print(f"  {i:2d}. {self.classes[str(i)]}", end='  ')
            if i % 6 == 0:
                print()
        
        print("\n\n🔢 NUMBERS (0-9):")
        for i in range(27, 37):
            print(f"  {i:2d}. {self.classes[str(i)]}", end='  ')
        
        print("\n\n💬 COMMON PHRASES:")
        for i in range(37, 52):
            print(f"  {i:2d}. {self.classes[str(i)]}")
        
        print("\n" + "-" * 40)
        
        # Get user choice
        while True:
            try:
                choice = input("\nEnter class number (1-51): ").strip()
                if choice in self.classes:
                    class_name = self.classes[choice]
                    print(f"\n✅ Selected: {class_name}")
                    return class_name
                else:
                    print("❌ Invalid choice. Please enter a number between 1 and 51.")
            except:
                print("❌ Invalid input. Please enter a number.")
    
    def collect_class_data(self, class_name, num_samples=100):
        """Automatically collect data for a specific class"""
        print(f"\n{'='*60}")
        print(f"📸 AUTO-COLLECTING: {class_name}")
        print(f"{'='*60}")
        
        # Create class directories
        raw_class_dir = os.path.join(self.raw_dir, class_name)
        processed_class_dir = os.path.join(self.processed_dir, class_name)
        os.makedirs(raw_class_dir, exist_ok=True)
        os.makedirs(processed_class_dir, exist_ok=True)
        
        # Check existing count
        existing_raw = len([f for f in os.listdir(raw_class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        existing_processed = len([f for f in os.listdir(processed_class_dir) if f.endswith('.npy')])
        
        print(f"\n📊 Existing data:")
        print(f"   Raw images: {existing_raw}")
        print(f"   Processed landmarks: {existing_processed}")
        
        if existing_processed >= num_samples:
            print(f"✅ Already have {existing_processed} samples for {class_name}")
            return True
        
        samples_needed = num_samples - existing_processed
        print(f"\n🎯 Need to collect: {samples_needed} samples")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n📝 INSTRUCTIONS:")
        print(f"   1. Show the sign for '{class_name}' clearly")
        print("   2. Hold steady for 2 seconds before collection starts")
        print("   3. Move hand slightly between captures for variety")
        print("   4. Press SPACE to pause/resume")
        print("   5. Press ESC to stop early")
        print("\n⏳ Starting in 3 seconds...")
        
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        count = existing_processed
        capture_count = 0
        paused = False
        steady_frames = 0
        last_capture_time = time.time()
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            # Draw hand landmarks if detected
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    )
            
            # Add overlay
            cv2.putText(frame, f"Class: {class_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Progress: {count}/{num_samples}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not paused else (0, 165, 255), 2)
            
            if hand_detected:
                cv2.putText(frame, "✓ HAND DETECTED", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "✗ NO HAND", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if paused:
                cv2.putText(frame, "⏸ PAUSED", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                cv2.putText(frame, f"▶ Collecting...", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, "SPACE: Pause | ESC: Stop", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f'Collecting: {class_name}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE - pause
                paused = not paused
                if paused:
                    print("⏸ Paused")
                else:
                    print("▶ Resumed")
            
            elif key == 27:  # ESC - stop
                print("\n⏹ Stopped early by user")
                break
            
            # Auto-capture logic
            if not paused and hand_detected and count < num_samples:
                steady_frames += 1
                
                # Capture after 10 steady frames (about 0.3 seconds)
                if steady_frames > 10 and (time.time() - last_capture_time) > 0.5:
                    # Save raw image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    raw_filename = os.path.join(raw_class_dir, f"{class_name}_{timestamp}.jpg")
                    cv2.imwrite(raw_filename, frame)
                    
                    # Extract and save landmarks
                    landmarks = []
                    for hand_landmarks in results.multi_hand_landmarks[0].landmark:
                        landmarks.extend([hand_landmarks.x, hand_landmarks.y, hand_landmarks.z])
                    
                    # Ensure 63 features (21 landmarks * 3 coordinates)
                    if len(landmarks) >= 63:
                        landmarks = landmarks[:63]
                    else:
                        landmarks.extend([0] * (63 - len(landmarks)))
                    
                    landmarks_filename = os.path.join(processed_class_dir, f"{class_name}_{timestamp}.npy")
                    np.save(landmarks_filename, np.array(landmarks))
                    
                    count += 1
                    capture_count += 1
                    last_capture_time = time.time()
                    steady_frames = 0
                    
                    print(f"  ✓ Captured {count}/{num_samples} (Total: {capture_count})")
            else:
                steady_frames = 0
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Collection complete for {class_name}!")
        print(f"   Total captured: {count} samples")
        print(f"   Saved in: {processed_class_dir}")
        
        return count >= num_samples
    
    def run(self):
        """Main collection loop"""
        print("="*60)
        print("🤟 ISL AUTO DATA COLLECTOR")
        print("="*60)
        
        while True:
            # Select class
            class_name = self.select_class()
            
            # Ask for number of samples
            try:
                num_samples = input("\nHow many samples to collect? (default: 100): ").strip()
                num_samples = int(num_samples) if num_samples else 100
            except:
                num_samples = 100
                print("Using default: 100 samples")
            
            # Collect data
            success = self.collect_class_data(class_name, num_samples)
            
            if success:
                print(f"\n✅ Successfully collected {class_name}")
            else:
                print(f"\n⚠️ Collection incomplete for {class_name}")
            
            # Ask to continue
            cont = input("\nCollect another class? (y/n): ").strip().lower()
            if cont != 'y':
                break
        
        print("\n" + "="*60)
        print("📊 COLLECTION SUMMARY")
        print("="*60)
        
        # Show final stats
        print("\n📁 Dataset folders:")
        for class_name in sorted(os.listdir(self.processed_dir)):
            class_path = os.path.join(self.processed_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.endswith('.npy')])
                print(f"   {class_name}: {count} samples")
        
        print("\n" + "="*60)
        print("✅ Ready for training!")
        print("="*60)

if __name__ == "__main__":
    collector = AutoDataCollector()
    collector.run()