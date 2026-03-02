import cv2
import os

def create_folder(folder_path):
    """Create folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

def collect_data():
    # Initialize webcam once
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set a smaller frame size for faster processing (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Ask for the class label
        class_name = input("\nEnter class name (e.g., A, B, 1, 2, ...) or 'quit' to exit: ").strip()
        if class_name.lower() == 'quit':
            break
        if not class_name:
            print("Class name cannot be empty. Try again.")
            continue

        # Create directory for this class
        save_dir = os.path.join("data", class_name)
        create_folder(save_dir)

        # Counter for naming images
        img_counter = 0

        print(f"\nCollecting images for class '{class_name}'.")
        print("Press 's' to save the current frame, 'q' to finish this class.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Draw a rectangle to guide hand placement (optional)
            cv2.rectangle(frame, (300, 200), (500, 400), (0, 255, 0), 2)
            cv2.putText(frame, f"Class: {class_name} | Images: {img_counter}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Data Collection - Press 's' to save, 'q' to finish class", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                img_name = f"image_{img_counter:03d}.jpg"
                img_path = os.path.join(save_dir, img_name)
                cv2.imwrite(img_path, frame)
                print(f"Saved: {img_path}")
                img_counter += 1

            elif key == ord('q'):
                print(f"Finished collecting for class '{class_name}'. Total images: {img_counter}")
                break

        # After finishing a class, ask if user wants to continue with another
        again = input("\nDo you want to collect another class? (y/n): ").strip().lower()
        if again != 'y':
            break

    # Release resources after all classes are done
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection completed.")

if __name__ == "__main__":
    collect_data()