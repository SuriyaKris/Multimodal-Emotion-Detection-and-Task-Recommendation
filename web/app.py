import os
import cv2
import numpy as np
import tensorflow as tf
import librosa
import pickle
import sounddevice as sd
import queue
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.io.wavfile import write

app = Flask(__name__)

# Load trained models
text_model = tf.keras.models.load_model("models/text_emotion_model.h5")
speech_model = tf.keras.models.load_model("models/speech_emotion_model.h5")
facial_model = tf.keras.models.load_model("models/facial_emotion_model.h5")

# Load tokenizer
with open("data/processed/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define emotion classes
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised', 'surprised']

# Activity suggestions
activity_suggestions = {
    "happy": "Listen to music or go for a walk!",
    "sad": "Watch a feel-good movie or call a friend.",
    "angry": "Try deep breathing exercises or go for a jog.",
    "fearful": "Read a book or do meditation.",
    "neutral": "Continue with your daily tasks or try something new!",
    "surprised": "Celebrate the moment or share the experience with someone!",
    "calm": "Enjoy the peaceful moment or do some yoga.",
    "disgust": "Try shifting your focus to something pleasant."
}

# Function to predict text emotion
def predict_text_emotion(text):
    max_len = 100
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = text_model.predict(padded_sequence)
    emotion = emotion_classes[np.argmax(prediction)]
    return emotion, activity_suggestions.get(emotion, "No suggestion available.")

# Function to record and process speech
q = queue.Queue()
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def record_audio(filename="web/static/realtime_audio.wav", duration=15, samplerate=22050):
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        print("Recording...")
        audio_data = []
        for _ in range(int(samplerate / 1024 * duration)):
            audio_data.append(q.get())
        audio_array = np.concatenate(audio_data, axis=0)
        write(filename, samplerate, audio_array)
    print("Recording complete.")

# Function to predict speech emotion
def predict_speech_emotion():
    record_audio()
    audio, sr = librosa.load("web/static/realtime_audio.wav", sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    expected_length = speech_model.input_shape[2]  
    current_length = mfccs.shape[1]

    if current_length < expected_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, expected_length - current_length)), mode='constant')
    elif current_length > expected_length:
        mfccs = mfccs[:, :expected_length]

    mfccs = np.expand_dims(mfccs, axis=[-1, 0])  

    prediction = speech_model.predict(mfccs)
    emotion = emotion_classes[np.argmax(prediction)]
    return emotion, activity_suggestions.get(emotion, "No suggestion available.")

# Function to capture and predict facial emotion
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
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48)) / 255.0
    reshaped = resized.reshape(1, 48, 48, 1)
    
    prediction = facial_model.predict(reshaped)
    emotion = emotion_classes[np.argmax(prediction)]
    return emotion, activity_suggestions.get(emotion, "No suggestion available.")

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_text", methods=["POST"])
def predict_text():
    text = request.form["text"]
    emotion, suggestion = predict_text_emotion(text)
    return jsonify({"emotion": emotion, "suggestion": suggestion})

@app.route("/predict_speech", methods=["POST"])
def predict_speech():
    emotion, suggestion = predict_speech_emotion()
    return jsonify({"emotion": emotion, "suggestion": suggestion})

@app.route("/predict_facial", methods=["POST"])
def predict_facial():
    emotion, suggestion = predict_facial_emotion()
    return jsonify({"emotion": emotion, "suggestion": suggestion})

if __name__ == "__main__":
    app.run(debug=True)
