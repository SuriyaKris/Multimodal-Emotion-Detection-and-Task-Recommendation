import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Define paths
data_path = "data/speech_emotion"  # Modify as needed
output_path = "data/processed/speech/"
os.makedirs(output_path, exist_ok=True)

# Function to extract MFCC features
def extract_mfcc(file_path, max_len=200):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs = np.pad(mfccs, ((0, 0), (0, max(0, max_len - mfccs.shape[1]))), mode='constant')
    return mfccs[:, :max_len]

# Process dataset
file_paths = []
labels = []
features = []

for root, _, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            label = int(file.split("-")[2]) - 1  # Extract emotion label from filename
            file_paths.append(file_path)
            labels.append(label)
            features.append(extract_mfcc(file_path))

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Save processed data
np.save(os.path.join(output_path, "X_train.npy"), X_train)
np.save(os.path.join(output_path, "X_test.npy"), X_test)
np.save(os.path.join(output_path, "y_train.npy"), y_train)
np.save(os.path.join(output_path, "y_test.npy"), y_test)

print("Speech preprocessing complete. Data saved.")
