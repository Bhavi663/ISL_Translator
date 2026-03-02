import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from collections import Counter
import time

class ISLModelTrainer:
    def __init__(self):
        self.base_dir = "dataset"
        self.processed_dir = os.path.join(self.base_dir, "processed_landmarks")
        self.models_dir = "models"
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Training parameters
        self.test_size = 0.15
        self.val_size = 0.15
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        
    def load_landmarks(self):
        """Load all processed landmarks from the dataset."""
        print("\n" + "="*60)
        print("📂 LOADING PROCESSED LANDMARKS")
        print("="*60)
        
        X = []
        y = []
        class_counts = {}
        
        # Get all class folders
        classes = [d for d in os.listdir(self.processed_dir) 
                  if os.path.isdir(os.path.join(self.processed_dir, d))]
        classes.sort()
        
        print(f"Found {len(classes)} classes with processed data")
        
        for class_name in classes:
            class_dir = os.path.join(self.processed_dir, class_name)
            landmarks_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            
            count = 0
            for lf in landmarks_files:
                try:
                    landmarks = np.load(os.path.join(class_dir, lf))
                    if len(landmarks) == 63:  # 21 landmarks * 3 coordinates
                        X.append(landmarks)
                        y.append(class_name)
                        count += 1
                except Exception as e:
                    print(f"  ⚠️ Error loading {lf}: {e}")
            
            class_counts[class_name] = count
            print(f"  {class_name}: {count} samples")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n✅ Total samples loaded: {len(X)}")
        print(f"✅ Feature dimension: {X.shape[1]}")
        
        # Show class distribution
        print("\n📊 Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / len(X)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        return X, y, classes
    
    def prepare_data(self, X, y):
        """Prepare data for training."""
        print("\n" + "="*60)
        print("🔧 PREPARING DATA")
        print("="*60)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Calculate normalization parameters
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        
        # Normalize data
        X_normalized = (X - mean) / std
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_normalized, y_encoded, 
            test_size=self.test_size,
            random_state=42,
            stratify=y_encoded
        )
        
        # Adjust validation size relative to remaining data
        val_relative_size = self.val_size / (1 - self.test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative_size,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"\n📈 Data split:")
        print(f"  Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Testing: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Convert to tensors
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, test_dataset, (X_train, y_train, X_val, y_val, X_test, y_test), mean, std, label_encoder
    
    def create_advanced_model(self, input_shape, num_classes):
        """Create an advanced neural network for better accuracy."""
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # First block - feature expansion
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second block
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third block
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth block
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train_model(self, train_dataset, val_dataset, input_shape, num_classes):
        """Train the model."""
        print("\n" + "="*60)
        print("🤖 TRAINING MODEL")
        print("="*60)
        
        # Create model
        model = self.create_advanced_model(input_shape, num_classes)
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel architecture:")
        model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.CSVLogger(os.path.join(self.models_dir, 'training_log.csv'))
        ]
        
        # Train
        print("\n🎯 Starting training...")
        start_time = time.time()
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        
        return model, history
    
    def evaluate_model(self, model, test_dataset, label_encoder):
        """Evaluate the model on test data."""
        print("\n" + "="*60)
        print("📊 EVALUATING MODEL")
        print("="*60)
        
        # Overall metrics
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
        print(f"\nTest accuracy: {test_accuracy*100:.2f}%")
        print(f"Test loss: {test_loss:.4f}")
        
        # Per-class metrics (optional - can be expanded)
        print("\n✅ Model evaluation complete")
        
        return test_accuracy, test_loss
    
    def save_model_artifacts(self, model, mean, std, label_encoder, classes):
        """Save model and all associated artifacts."""
        print("\n" + "="*60)
        print("💾 SAVING MODEL ARTIFACTS")
        print("="*60)
        
        # Save model
        model_path = os.path.join(self.models_dir, 'isl_model.keras')
        model.save(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save preprocessing data
        mean_path = os.path.join(self.models_dir, 'mean.npy')
        std_path = os.path.join(self.models_dir, 'std.npy')
        np.save(mean_path, mean)
        np.save(std_path, std)
        print(f"✓ Preprocessing data saved to: {self.models_dir}")
        
        # Save class mappings
        class_to_idx = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
        idx_to_class = {idx: cls for idx, cls in enumerate(label_encoder.classes_)}
        
        mapping_path = os.path.join(self.models_dir, 'class_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump(class_to_idx, f)
        print(f"✓ Class mapping saved to: {mapping_path}")
        
        # Save class list
        classes_path = os.path.join(self.models_dir, 'classes.txt')
        with open(classes_path, 'w') as f:
            for cls in label_encoder.classes_:
                f.write(f"{cls}\n")
        print(f"✓ Class list saved to: {classes_path}")
        
        return model_path
    
    def plot_training_history(self, history):
        """Plot and save training history."""
        print("\n📈 Plotting training history...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'], linewidth=2)
            axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Final accuracy comparison
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        axes[1, 1].bar(['Train', 'Validation'], [final_train_acc, final_val_acc], 
                       color=['#3498db', '#e74c3c'], alpha=0.7)
        axes[1, 1].set_title(f'Final Accuracy\nTrain: {final_train_acc:.3f} | Val: {final_val_acc:.3f}', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim([0, 1])
        
        # Add value labels on bars
        for i, v in enumerate([final_train_acc, final_val_acc]):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        history_path = os.path.join(self.models_dir, 'training_history.png')
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {history_path}")
        
        plt.show()
    
    def run_training(self):
        """Main training pipeline."""
        print("\n" + "="*60)
        print("🤟 ISL MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        X, y, classes = self.load_landmarks()
        
        if len(X) == 0:
            print("❌ No data found! Please collect data first using collect_data.py")
            return
        
        # Step 2: Prepare data
        train_dataset, val_dataset, test_dataset, data_arrays, mean, std, label_encoder = self.prepare_data(X, y)
        X_train, y_train, X_val, y_val, X_test, y_test = data_arrays
        
        # Step 3: Train model
        model, history = self.train_model(
            train_dataset, 
            val_dataset, 
            input_shape=X.shape[1],
            num_classes=len(classes)
        )
        
        # Step 4: Evaluate model
        test_accuracy, test_loss = self.evaluate_model(model, test_dataset, label_encoder)
        
        # Step 5: Save model artifacts
        model_path = self.save_model_artifacts(model, mean, std, label_encoder, classes)
        
        # Step 6: Plot training history
        self.plot_training_history(history)
        
        # Step 7: Final summary
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETE!")
        print("="*60)
        print(f"\n📊 Final Results:")
        print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Number of Classes: {len(classes)}")
        print(f"  Total Training Samples: {len(X_train)}")
        print(f"  Model saved to: {model_path}")
        print("\n✅ Your model is ready for use in the web app!")

if __name__ == "__main__":
    trainer = ISLModelTrainer()
    
    # Allow user to adjust parameters
    print("\n⚙️ Training Parameters:")
    print(f"  Current epochs: {trainer.epochs}")
    print(f"  Current batch size: {trainer.batch_size}")
    print(f"  Current learning rate: {trainer.learning_rate}")
    
    adjust = input("\nAdjust parameters? (y/n): ").strip().lower()
    if adjust == 'y':
        try:
            trainer.epochs = int(input("  Epochs (recommended: 100): "))
            trainer.batch_size = int(input("  Batch size (recommended: 32): "))
            trainer.learning_rate = float(input("  Learning rate (recommended: 0.001): "))
        except:
            print("  Using default values")
    
    trainer.run_training()