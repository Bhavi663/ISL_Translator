#!/usr/bin/env python3
"""
Batch Evaluation Script
----------------------
Evaluate model on entire validation dataset
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tensorflow as tf
    print("✅ TensorFlow imported")
except ImportError:
    print("❌ TensorFlow not found")
    sys.exit(1)

class BatchEvaluator:
    def __init__(self, model_folder):
        self.model = None
        self.load_model(model_folder)
    
    def load_model(self, model_folder):
        """Load model and preprocessors"""
        model_path = os.path.join(model_folder, "isl_model.keras")
        mean_path = os.path.join(model_folder, "mean.npy")
        std_path = os.path.join(model_folder, "std.npy")
        mapping_path = os.path.join(model_folder, "class_mapping.pkl")
        
        self.model = tf.keras.models.load_model(model_path)
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        
        with open(mapping_path, 'rb') as f:
            self.class_to_idx = pickle.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        print(f"✅ Model loaded: {len(self.class_to_idx)} classes")
    
    def load_validation_data(self, data_folder):
        """Load validation data from numpy files"""
        X_val = []
        y_val = []
        
        for class_name in os.listdir(data_folder):
            class_path = os.path.join(data_folder, class_name)
            if not os.path.isdir(class_path):
                continue
            
            class_idx = self.class_to_idx.get(class_name)
            if class_idx is None:
                continue
            
            for file in os.listdir(class_path):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(class_path, file))
                    X_val.append(data)
                    y_val.append(class_idx)
        
        return np.array(X_val), np.array(y_val)
    
    def evaluate(self, X_val, y_val):
        """Run evaluation"""
        print(f"\n📊 Evaluating on {len(X_val)} samples...")
        
        # Normalize
        X_norm = (X_val - self.mean) / self.std
        
        # Predict
        y_pred_proba = self.model.predict(X_norm, batch_size=32, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Generate report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        target_names = [self.idx_to_class[i] for i in range(len(self.class_to_idx))]
        print(classification_report(y_val, y_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names[:30] if len(target_names) > 30 else target_names,
                   yticklabels=target_names[:30] if len(target_names) > 30 else target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        print("\n✅ Confusion matrix saved to: confusion_matrix.png")

def main():
    evaluator = BatchEvaluator("models/trained_20250226_155216")
    X_val, y_val = evaluator.load_validation_data("path/to/validation/data")
    evaluator.evaluate(X_val, y_val)

if __name__ == "__main__":
    main()