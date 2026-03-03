import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# Load images from dataset directories
dataset_dir = "dataset"
parasitized_dir = os.path.join(dataset_dir, "Parasitized")
uninfected_dir = os.path.join(dataset_dir, "Uninfected")

def load_images_from_folder(folder_path, label):
    """Load and preprocess images from a folder"""
    images = []
    labels = []
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    return images, labels

# Load images
print("Loading parasitized images...")
parasitized_images, parasitized_labels = load_images_from_folder(parasitized_dir, 1)

print("Loading uninfected images...")
uninfected_images, uninfected_labels = load_images_from_folder(uninfected_dir, 0)

# Combine datasets
X = np.array(parasitized_images + uninfected_images)
y = np.array(parasitized_labels + uninfected_labels)

print(f"Total images loaded: {len(X)}")

# Normalize pixel values
X = X.astype('float32') / 255.0

# Split dataset (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save preprocessed data
processed_data_dir = "processed_data"
os.makedirs(processed_data_dir, exist_ok=True)

np.save(os.path.join(processed_data_dir, "X_train.npy"), X_train)
np.save(os.path.join(processed_data_dir, "X_val.npy"), X_val)
np.save(os.path.join(processed_data_dir, "y_train.npy"), y_train)
np.save(os.path.join(processed_data_dir, "y_val.npy"), y_val)

print(f"\nData saved to {processed_data_dir}/")
print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")
print("Data preprocessing completed!")
