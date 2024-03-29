import tkinter as tk
from tkinter import filedialog
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import os
import pandas as pd


emotion = {
    1:"neutral",
    2:"calm",
    3:"happy",
    4:"sad",
    5:"angry",
    6:"fearful",
    7:"disgust",
    8:"surprised"
}


def neural_network(data):
    model = tf.keras.models.load_model("model")
    predict=model.predict(data)
    return emotion[predict]

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

        #Spectral centroid
    sc = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, sc))

    #Spectral Spread
    ss = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, ss))

    # Spectral Flux
    sf = np.mean(librosa.onset.onset_strength(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, sf))

    # Spectral Roll-Off
    srf = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, srf))

    # Chroma Vector
    cv = np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, cv))

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    result=pd.DataFrame(result)
    return result

def process_audio_file(filepath):
    y, sr = librosa.load(filepath, duration=2.5, offset=0.6)

    features = extract_features(y, sr)
    print(features.shape) #CO SIE KURWA NIE ZGADZA
    output = neural_network(features)

    return output

def browse_file():
    filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
    if filepath:
        result = process_audio_file(filepath)
        result_label.config(text=f"Output: {result}")

# Tworzenie głównego okna
root = tk.Tk()
root.title("Audio Feature Extraction")

# Przycisk do wyboru pliku
browse_button = tk.Button(root, text="Wybierz plik", command=browse_file)
browse_button.pack(pady=20)

# Etykieta na wynik
result_label = tk.Label(root, text="Output: ")
result_label.pack()

# Uruchomienie pętli głównej aplikacji
root.mainloop()
