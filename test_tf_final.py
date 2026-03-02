import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    print(f"\n✅ TensorFlow imported successfully")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow location: {tf.__file__}")
    
    # Test basic functionality
    import numpy as np
    a = tf.constant([[1, 2], [3, 4]])
    print(f"\nBasic tensor operation works: {a}")
    
    # Test keras
    print(f"\nKeras version: {tf.keras.__version__}")
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print("✅ Can create Keras models")
    
except Exception as e:
    print(f"\n❌ Error: {e}")