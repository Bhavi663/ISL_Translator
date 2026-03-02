import cv2
import os
import mediapipe as mp
import numpy as np
from datetime import datetime
import time

class ISLDataCollector:
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
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Define all classes to collect
        self.classes = []
        
    def setup_classes(self):
        """Define all the signs to collect."""
        print("\n" + "="*60)
        print("📋 SETUP SIGN CLASSES")
        print("="*60)
        
        # Alphabets A-Z
        alphabets = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # Numbers 0-9
        numbers = [str(i) for i in range(10)]
        
        # Common words/phrases
        words = [
            "Hello", "Namaste", "Thank_You", "Please", "Sorry",
            "Yes", "No", "Help", "India", "Student", "Teacher",
            "Father", "Mother", "Brother", "Sister", "Friend",
            "Good_Morning", "Good_Night", "How_are_you", "I_am_fine",
            "What", "Where", "Why", "When", "Who", "Which",
            "Eat", "Drink", "Sleep", "Work", "Play", "Learn",
            "Happy", "Sad", "Angry", "Love", "Like", "Dislike"
        ]
        
        # Combine all classes
        self.classes = alphabets + numbers + words
        
        print(f"\nTotal classes to collect: {len(self.classes)}")
        print("\nClasses breakdown:")
        print(f"  - Alphabets (A-Z): {len(alphabets)} classes")
        print(f"  - Numbers (0-9): {len(numbers)} classes")
        print(f"  - Words/Phrases: {len(words)} classes")
        
        return self.classes
    
    def select_classes_to_collect(self):
        """Let user select which classes to collect."""
        print("\n" + "="*60)
        print("🎯 SELECT CLASSES TO COLLECT")
        print("="*60)
        print("\nOptions:")
        print("1. Collect all classes")
        print("2. Collect specific categories")
        print("3. Collect specific letters/numbers")
        print("4. Continue from last session")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            return self.classes
        elif choice == '2':
            return self.select_categories()
        elif choice == '3':
            return self.select_specific()
        elif choice == '4':
            return self.continue_last_session()
        else:
            print("Invalid choice. Collecting all classes.")
            return self.classes
    
    def select_categories(self):
        """Select classes by category."""
        print("\nCategories:")
        print("1. Alphabets only (A-Z)")
        print("2. Numbers only (0-9)")
        print("3. Words only")
        print("4. Custom combination")
        
        cat_choice = input("\nSelect category: ").strip()
        
        selected = []
        if cat_choice == '1':
            selected = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        elif cat_choice == '2':
            selected = [str(i) for i in range(10)]
        elif cat_choice == '3':
            words = [
                "Hello", "Namaste", "Thank_You", "Please", "Sorry",
                "Yes", "No", "Help", "India", "Student"
            ]
            selected = words
        elif cat_choice == '4':
            # Let user type specific classes
            print("\nEnter classes separated by commas (e.g., A,B,C,Hello,Namaste):")
            custom = input().strip().split(',')
            selected = [c.strip() for c in custom]
        
        return selected
    
    def select_specific(self):
        """Select specific letters/numbers."""
        print("\nEnter specific signs (comma-separated, e.g., A,B,C,1,2,3,Hello):")
        specific = input().strip().split(',')
        return [s.strip() for s in specific]
    
    def continue_last_session(self):
        """Continue from last incomplete session."""
        completed = self.get_completed_classes()
        remaining = [c for c in self.classes if c not in completed]
        print(f"\nCompleted classes: {len(completed)}")
        print(f"Remaining classes: {len(remaining)}")
        
        if remaining:
            print("\nContinuing with remaining classes...")
            return remaining
        else:
            print("All classes completed! Starting fresh...")
            return self.classes
    
    def get_completed_classes(self):
        """Get list of classes that already have data."""
        completed = []
        for class_name in self.classes:
            class_dir = os.path.join(self.raw_dir, class_name)
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))]
                if len(images) >= 100:  # Minimum images per class
                    completed.append(class_name)
        return completed
    
    def collect_sign_data(self, class_name, target_count=100):
        """Collect images for a specific sign."""
        print(f"\n{'='*60}")
        print(f"📸 COLLECTING: {class_name}")
        print(f"{'='*60}")
        
        # Create class directories
        raw_class_dir = os.path.join(self.raw_dir, class_name)
        processed_class_dir = os.path.join(self.processed_dir, class_name)
        os.makedirs(raw_class_dir, exist_ok=True)
        os.makedirs(processed_class_dir, exist_ok=True)
        
        # Check existing count
        existing = len([f for f in os.listdir(raw_class_dir) 
                       if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if existing >= target_count:
            print(f"✅ Already have {existing} images for {class_name}")
            return True
        
        print(f"Need {target_count - existing} more images for {class_name}")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n📝 Instructions:")
        print(f"  - Show the sign for '{class_name}'")
        print("  - Press SPACE to capture an image")
        print("  - Press 'r' to restart counting")
        print("  - Press ESC to skip this sign")
        print("  - Press 'q' to quit completely")
        print("\nPress any key to start...")
        cv2.waitKey(0)
        
        count = existing
        capturing = True
        
        while capturing and count < target_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    )
            
            # Add overlay text
            cv2.putText(frame, f"Sign: {class_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{target_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                cv2.putText(frame, "HAND DETECTED ✓", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO HAND DETECTED ✗", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, "SPACE: Capture | R: Reset | ESC: Skip | Q: Quit", 
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f"Collecting: {class_name}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE - capture
                # Save raw image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_filename = os.path.join(raw_class_dir, f"{class_name}_{timestamp}.jpg")
                cv2.imwrite(raw_filename, frame)
                
                # Save landmarks if detected
                if results.multi_hand_landmarks:
                    landmarks = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                    
                    # Pad or truncate to fixed length (21 landmarks * 3 coordinates = 63)
                    if len(landmarks) < 63:
                        landmarks.extend([0] * (63 - len(landmarks)))
                    else:
                        landmarks = landmarks[:63]
                    
                    landmarks_filename = os.path.join(processed_class_dir, 
                                                      f"{class_name}_{timestamp}.npy")
                    np.save(landmarks_filename, np.array(landmarks))
                    
                    count += 1
                    print(f"  ✓ Captured {count}/{target_count}")
                else:
                    print("  ✗ No hand detected - image saved but landmarks not extracted")
            
            elif key == ord('r'):  # R - reset
                count = existing
                print(f"  Reset to {count}")
            
            elif key == 27:  # ESC - skip this sign
                print(f"  Skipping {class_name}")
                capturing = False
            
            elif key == ord('q'):  # Q - quit
                print("  Quitting...")
                capturing = False
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Completed {class_name}: {count} images captured")
        return True
    
    def run_collection(self):
        """Main collection loop."""
        print("\n" + "="*60)
        print("🤟 ISL SIGN LANGUAGE DATA COLLECTOR")
        print("="*60)
        
        # Setup classes
        self.setup_classes()
        
        # Select which classes to collect
        classes_to_collect = self.select_classes_to_collect()
        
        if not classes_to_collect:
            print("No classes selected. Exiting.")
            return
        
        print(f"\n📊 Will collect data for {len(classes_to_collect)} classes:")
        for i, cls in enumerate(classes_to_collect[:10], 1):
            print(f"  {i}. {cls}")
        if len(classes_to_collect) > 10:
            print(f"  ... and {len(classes_to_collect) - 10} more")
        
        # Ask for images per class
        try:
            target = int(input("\nHow many images per class? (recommended: 100-200): ").strip())
        except:
            target = 100
            print(f"Using default: {target} images per class")
        
        # Collect data
        successful = []
        failed = []
        
        for i, class_name in enumerate(classes_to_collect, 1):
            print(f"\n{'='*60}")
            print(f"Progress: {i}/{len(classes_to_collect)}")
            print(f"{'='*60}")
            
            result = self.collect_sign_data(class_name, target)
            
            if result:
                successful.append(class_name)
            else:
                failed.append(class_name)
            
            # Ask to continue
            if i < len(classes_to_collect):
                cont = input("\nContinue to next sign? (y/n): ").strip().lower()
                if cont != 'y':
                    print("Collection paused.")
                    break
        
        # Summary
        print("\n" + "="*60)
        print("📊 COLLECTION SUMMARY")
        print("="*60)
        print(f"Successful: {len(successful)} classes")
        print(f"Failed/Skipped: {len(failed)} classes")
        
        if successful:
            print("\nCompleted classes:")
            for cls in successful:
                print(f"  ✓ {cls}")
        
        if failed:
            print("\nIncomplete classes:")
            for cls in failed:
                print(f"  ✗ {cls}")
        
        print("\n✅ Data collection complete!")

if __name__ == "__main__":
    collector = ISLDataCollector()
    collector.run_collection()