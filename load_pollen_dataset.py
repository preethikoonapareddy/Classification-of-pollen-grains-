import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_images_from_pollen_folder(path='pollen', image_size=(128, 128)):
    X = []
    Y = []
    for file in os.listdir(path):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            label = file.replace(' ', '_').split('_')[0].lower()
            img_path = os.path.join(path, file)
            try:
                img = Image.open(img_path).convert('RGB').resize(image_size)
                X.append(np.array(img) / 255.0)
                Y.append(label)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
    return np.array(X), np.array(Y)

# Load and preprocess data
X, Y = load_images_from_pollen_folder('pollen')
print(f"Loaded {len(X)} images.")

# Encode labels
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
Y_classes = tf.keras.utils.to_categorical(Y_encoded, num_classes=len(np.unique(Y_encoded)))

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_classes, test_size=0.25, random_state=42, stratify=Y_encoded)

# Save label mapping
with open("class_labels.txt", "w") as f:
    for label in label_encoder.classes_:
        f.write(label + "\n")