import os
import cv2
import numpy as np
import tensorflow as tf
import librosa
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sounddevice as sd
import wave
import queue
from scipy.io.wavfile import write

# Load trained models
text_model = tf.keras.models.load_model("models/text_emotion_model.h5")
speech_model = tf.keras.models.load_model("models/speech_emotion_model.h5")
facial_model = tf.keras.models.load_model("models/facial_emotion_model.h5")

# Load tokenizer
with open("data/processed/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define emotion classes manually
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised', 'surprised']

# Suggested activities based on emotion
def suggest_activity(emotion):
    suggestions = {
        'neutral': "Continue with your regular tasks or take a short break.",
        'calm': "Try meditation or enjoy a peaceful walk.",
        'happy': "Engage in creative work or spend time with friends.",
        'sad': "Listen to uplifting music or talk to a loved one.",
        'angry': "Take deep breaths, go for a run, or write down your thoughts.",
        'fearful': "Practice relaxation techniques or talk to someone you trust.",
        'disgust': "Distract yourself with a fun activity or engage in a hobby.",
        'surprised': "Use the energy to explore new ideas or take on a challenge."
    }
    return suggestions.get(emotion, "No specific suggestion available.")

# Function for real-time text emotion prediction
def predict_text_emotion(text):
    max_len = 100
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = text_model.predict(padded_sequence)
    detected_emotion = emotion_classes[np.argmax(prediction)]
    return detected_emotion, suggest_activity(detected_emotion)

# Function to record audio
q = queue.Queue()
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def record_audio(filename="realtime_audio.wav", duration=15, samplerate=22050):
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        print("Recording...")
        audio_data = []
        for _ in range(int(samplerate / 1024 * duration)):
            audio_data.append(q.get())
        audio_array = np.concatenate(audio_data, axis=0)
        write(filename, samplerate, audio_array)
    print("Recording complete.")

# Function for real-time speech emotion prediction
def predict_speech_emotion():
    record_audio()
    audio, sr = librosa.load("realtime_audio.wav", sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Ensure the MFCC shape matches model expectation
    expected_length = 800  # Adjust based on training
    current_length = mfccs.shape[1]
    
    if current_length < expected_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, expected_length - current_length)), mode='constant')
    elif current_length > expected_length:
        mfccs = mfccs[:, :expected_length]
    
    mfccs = np.expand_dims(mfccs, axis=[-1, 0])  # Reshape for model
    prediction = speech_model.predict(mfccs)
    detected_emotion = emotion_classes[np.argmax(prediction)]
    return detected_emotion, suggest_activity(detected_emotion)

# Function for real-time facial emotion prediction
def predict_facial_emotion():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to capture image")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    # Preprocess image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48)) / 255.0
    reshaped = resized.reshape(1, 48, 48, 1)
    prediction = facial_model.predict(reshaped)
    detected_emotion = emotion_classes[np.argmax(prediction)]
    return detected_emotion, suggest_activity(detected_emotion)

# Main function to get real-time input
if __name__ == "__main__":
    while True:
        print("Choose input mode: [1] Text, [2] Speech, [3] Facial, [q] Quit")
        choice = input("Enter choice: ")
        if choice == '1':
            text = input("Enter text: ")
            emotion, suggestion = predict_text_emotion(text)
            print(f"Detected Emotion: {emotion}\nSuggested Activity: {suggestion}")
        elif choice == '2':
            emotion, suggestion = predict_speech_emotion()
            print(f"Detected Emotion: {emotion}\nSuggested Activity: {suggestion}")
        elif choice == '3':
            emotion, suggestion = predict_facial_emotion()
            print(f"Detected Emotion: {emotion}\nSuggested Activity: {suggestion}")
        elif choice.lower() == 'q':
            break
        else:
            print("Invalid choice, try again.")
