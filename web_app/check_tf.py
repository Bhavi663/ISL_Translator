import sys 
print(f"Python version: {sys.version}") 
print(f"Python executable: {sys.executable}") 
 
try: 
    import tensorflow as tf 
    print(f"? TensorFlow imported successfully") 
    print(f"TensorFlow version: {tf.__version__}") 
    print(f"TensorFlow location: {tf.__file__}") 
except ImportError as e: 
    print(f"? Cannot import tensorflow: {e}") 
except AttributeError as e: 
    print(f"? TensorFlow has no version attribute: {e}") 
    import tensorflow as tf 
    print(f"Dir of tensorflow: {dir(tf)[:20]}") 
