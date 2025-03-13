import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
data_path = "data/FER-2013/"  # Modify as needed
output_path = "data/processed/facial/"
os.makedirs(output_path, exist_ok=True)

# Function to preprocess image
def process_image(image_path, img_size=(48, 48)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, img_size)  # Resize to 48x48
    image = image / 255.0  # Normalize
    return image

# Load dataset
categories = os.listdir(os.path.join(data_path, "train"))
X_train, y_train, X_test, y_test = [], [], [], []

# Process training data
for label, category in enumerate(categories):
    train_folder = os.path.join(data_path, "train", category)
    for img_name in os.listdir(train_folder):
        img_path = os.path.join(train_folder, img_name)
        X_train.append(process_image(img_path))
        y_train.append(label)

# Process test data
for label, category in enumerate(categories):
    test_folder = os.path.join(data_path, "test", category)
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        X_test.append(process_image(img_path))
        y_test.append(label)

# Convert to NumPy arrays
X_train = np.array(X_train).reshape(-1, 48, 48, 1)
X_test = np.array(X_test).reshape(-1, 48, 48, 1)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Save processed data
np.save(os.path.join(output_path, "X_train.npy"), X_train)
np.save(os.path.join(output_path, "X_test.npy"), X_test)
np.save(os.path.join(output_path, "y_train.npy"), y_train)
np.save(os.path.join(output_path, "y_test.npy"), y_test)

print("Facial image preprocessing complete. Data saved.")
