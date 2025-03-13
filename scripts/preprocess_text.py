import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required resources
nltk.download('stopwords')
nltk.download('punkt')

# Load text dataset (modify path accordingly)
data_path = "data/text_emotion/emotion_dataset_raw.csv"
df = pd.read_csv(data_path)

# Text preprocessing function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = df['Text'].apply(clean_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['Emotion'], test_size=0.2, random_state=42)

# Tokenization
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

# Save preprocessed data
os.makedirs("data/processed", exist_ok=True)
pd.DataFrame({'text': X_train, 'label': y_train}).to_csv("data/processed/train_text.csv", index=False)
pd.DataFrame({'text': X_test, 'label': y_test}).to_csv("data/processed/test_text.csv", index=False)

# Save tokenizer
import pickle
with open("data/processed/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Text preprocessing complete. Tokenizer saved.")
