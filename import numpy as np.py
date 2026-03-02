import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# ===== CONFIGURATION =====
# Specify the 5 classes you want to analyse (use EXACT names as in your dataset)
selected_classes = ['A', 'B', 'C', 'D', 'E']  # <-- CHANGE THESE TO YOUR DESIRED CLASSES
# =========================

# Load the test data
X_test = np.load("dataset/X_test.npy")
y_test = np.load("dataset/y_test.npy")

# Load class mapping
with open("dataset/class_mapping.pkl", "rb") as f:
    class_to_idx = pickle.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Get the indices (integer labels) of the selected classes
selected_indices = [class_to_idx[cls] for cls in selected_classes if cls in class_to_idx]
if len(selected_indices) != len(selected_classes):
    missing = set(selected_classes) - set(class_to_idx.keys())
    print(f"Warning: These classes were not found in the dataset: {missing}")
    # Keep only those that exist
    selected_classes = [cls for cls in selected_classes if cls in class_to_idx]
    selected_indices = [class_to_idx[cls] for cls in selected_classes]

print(f"Analysing classes: {selected_classes}")
print(f"Corresponding indices: {selected_indices}")

# Create a boolean mask for samples that belong to any of the selected classes
mask = np.isin(y_test, selected_indices)

# Filter the data
X_test_filtered = X_test[mask]
y_test_filtered = y_test[mask]

print(f"Number of test samples for selected classes: {len(y_test_filtered)}")

if len(y_test_filtered) == 0:
    print("No samples found for the selected classes. Exiting.")
    exit()

# Load the trained model (optional: if you want fresh predictions)
# Alternatively, you can use saved predictions if you have them.
model = tf.keras.models.load_model("models/isl_model.keras")
print("Model loaded.")

# Predict on the filtered test set
y_pred_probs = model.predict(X_test_filtered)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test_filtered, y_pred, labels=selected_indices)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix for {len(selected_classes)} Classes')
plt.tight_layout()
plt.savefig('confusion_matrix_5classes.png', dpi=150)
plt.show()

# Optional: Print classification report for these classes
print("\nClassification Report for Selected Classes:")
print(classification_report(y_test_filtered, y_pred, 
                            target_names=selected_classes, labels=selected_indices))