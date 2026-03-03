import numpy as np
import matplotlib.pyplot as plt
import os

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.optimizers import Adam
except ImportError as e:
    print("Error: TensorFlow is not installed or virtual environment is not activated.")
    print("Please activate the virtual environment first:")
    print("  .\\tf_env\\Scripts\\Activate.ps1")
    print("Then install dependencies:")
    print("  python -m pip install tensorflow numpy matplotlib scikit-learn Pillow")
    exit(1)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load preprocessed data
print("Loading preprocessed data...")
processed_data_dir = "processed_data"

try:
    X_train = np.load(os.path.join(processed_data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_data_dir, "y_train.npy"))
    
    # Try to load X_test.npy and y_test.npy, if not found use X_val and y_val
    if os.path.exists(os.path.join(processed_data_dir, "X_test.npy")):
        X_test = np.load(os.path.join(processed_data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(processed_data_dir, "y_test.npy"))
        print("Loaded X_test.npy and y_test.npy")
    else:
        # Use validation set as test set
        X_test = np.load(os.path.join(processed_data_dir, "X_val.npy"))
        y_test = np.load(os.path.join(processed_data_dir, "y_val.npy"))
        print("Using X_val.npy and y_val.npy as test set")
    
    print(f"Data loaded successfully!")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
except FileNotFoundError as e:
    print(f"Error: Required data files not found. {e}")
    print("Please run preprocess_data.py first to generate the data files.")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Build CNN model: Conv2D → MaxPool → Dense
print("\nBuilding CNN model...")
try:
    model = Sequential([
        # Conv2D → MaxPool
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        
        # Conv2D → MaxPool
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Conv2D → MaxPool
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten before Dense layers
        Flatten(),
        
        # Dense layers
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model with Adam optimizer and binary_crossentropy
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model built successfully!")
    model.summary()
    
except Exception as e:
    print(f"Error building model: {e}")
    exit(1)

# Train model for 10 epochs
print("\nTraining model for 10 epochs...")
try:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Save model
print("\nSaving model...")
try:
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "custom_cnn_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
except Exception as e:
    print(f"Error saving model: {e}")
    exit(1)

# Plot accuracy and loss curves
print("\nPlotting training curves...")
try:
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Plot loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to training_curves.png")
    plt.show()
    
except Exception as e:
    print(f"Error plotting curves: {e}")

# Print final metrics
print("\n" + "="*50)
print("Training Summary")
print("="*50)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print("="*50)
print("\nTraining script completed successfully!")
