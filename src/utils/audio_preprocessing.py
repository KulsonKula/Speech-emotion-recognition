import numpy as np
import librosa
import pandas as pd

def extract_features(data, sample_rate):
    
    # ZCR
    features = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    features = np.hstack((features, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    features = np.hstack((features, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    features = np.hstack((features, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    features = np.hstack((features, rms))

        #Spectral centroid
    sc = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    features = np.hstack((features, sc))

    #Spectral Spread
    ss = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
    features = np.hstack((features, ss))

    # Spectral Flux
    sf = np.mean(librosa.onset.onset_strength(y=data, sr=sample_rate).T, axis=0)
    features = np.hstack((features, sf))

    # Spectral Roll-Off
    srf = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
    features = np.hstack((features, srf))

    # Chroma Vector
    cv = np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate).T, axis=0)
    features = np.hstack((features, cv))

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    features = np.hstack((features, mel))

    X = []
    X.append(features)
    Features = pd.DataFrame(X)
    X = Features.iloc[: ,:-1].values
    
    return X

def preprocess_audio(filepath):
    data, sample_rate = librosa.load(filepath, duration=2.5, offset=0.6)

    return data, sample_rate