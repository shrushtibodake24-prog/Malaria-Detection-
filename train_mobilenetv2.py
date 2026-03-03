import numpy as np
import matplotlib.pyplot as plt
import os

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
print("="*60)
print("Loading preprocessed data...")
print("="*60)
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
    
    print(f"\nData loaded successfully!")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Data type: {X_train.dtype}")
    print(f"  Value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    
except FileNotFoundError as e:
    print(f"Error: Required data files not found. {e}")
    print("Please run preprocess_data.py first to generate the data files.")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Build MobileNetV2 transfer learning model
print("\n" + "="*60)
print("Building MobileNetV2 Transfer Learning Model...")
print("="*60)

try:
    # Load MobileNetV2 base model
    # base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    print("Loading MobileNetV2 base model with ImageNet weights...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze all layers of base_model
    print("Freezing all base model layers...")
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False
    
    print(f"Number of trainable parameters in base model: {sum([tf.size(w).numpy() for w in base_model.trainable_weights])}")
    
    # Add custom layers on top
    print("Adding custom classification layers...")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print("Model architecture created successfully!")
    print("\nModel Summary:")
    model.summary()
    
except Exception as e:
    print(f"Error building model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Compile model
print("\n" + "="*60)
print("Compiling model...")
print("="*60)

try:
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled successfully!")
except Exception as e:
    print(f"Error compiling model: {e}")
    exit(1)

# Setup ImageDataGenerator with light augmentation
print("\n" + "="*60)
print("Setting up ImageDataGenerator with light augmentation...")
print("="*60)

# Light augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

# No augmentation for validation/test
test_datagen = ImageDataGenerator()

# Create generators
print("Creating data generators...")
train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=32,
    shuffle=True
)

test_generator = test_datagen.flow(
    X_test, y_test,
    batch_size=32,
    shuffle=False
)

print(f"Training batches per epoch: {len(train_generator)}")
print(f"Test batches: {len(test_generator)}")

# Train model for 10 epochs
print("\n" + "="*60)
print("Training model for 10 epochs...")
print("="*60)

try:
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=test_generator,
        validation_steps=len(test_generator),
        epochs=10,
        verbose=1
    )
    print("\nTraining completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Save model
print("\n" + "="*60)
print("Saving model...")
print("="*60)

try:
    # Save to models/mobilenetv2_model.h5 (standard location)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "mobilenetv2_model.h5")
    model.save(model_path)
    print(f"Model saved successfully to: {model_path}")
    
    # Verify model can be loaded
    print("Verifying model can be loaded...")
    loaded_model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully! Verification passed.")
    
except Exception as e:
    print(f"Error saving model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Plot accuracy and loss curves
print("\n" + "="*60)
print("Plotting training curves...")
print("="*60)

try:
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)
    plt.title('MobileNetV2 Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Plot loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    plt.title('MobileNetV2 Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mobilenetv2_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to mobilenetv2_training_curves.png")
    plt.show()
    
except Exception as e:
    print(f"Error plotting curves: {e}")
    import traceback
    traceback.print_exc()

# Print final metrics
print("\n" + "="*60)
print("Training Summary")
print("="*60)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print("="*60)

# Evaluate on test set
print("\n" + "="*60)
print("Evaluating on test set...")
print("="*60)

try:
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

print("\n" + "="*60)
print("MobileNetV2 Transfer Learning Training Completed Successfully!")
print("="*60)
print(f"Model saved to: {model_path}")
print(f"Training curves saved to: mobilenetv2_training_curves.png")
print("="*60)
