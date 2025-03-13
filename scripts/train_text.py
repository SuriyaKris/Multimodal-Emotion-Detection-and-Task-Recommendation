import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import pickle

# Define paths
data_path = "data/processed/"
output_path = "models/"
os.makedirs(output_path, exist_ok=True)

# Load data
df_train = pd.read_csv(os.path.join(data_path, "train_text.csv"))
df_test = pd.read_csv(os.path.join(data_path, "test_text.csv"))

X_train, y_train = df_train['text'].astype(str), df_train['label']
X_test, y_test = df_test['text'].astype(str), df_test['label']


# Load tokenizer
with open(os.path.join(data_path, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

# Tokenize and pad sequences
max_len = 100
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

# Convert labels to categorical
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Define LSTM model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    SpatialDropout1D(0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train_seq, y_train, epochs=10, batch_size=32, validation_data=(X_test_seq, y_test))

# Save model
model.save(os.path.join(output_path, "text_emotion_model.h5"))
print("Text emotion model training complete. Model saved.")
