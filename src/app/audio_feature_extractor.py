import tkinter as tk
import numpy as np
import pandas as pd
from tkinter import filedialog
from utils import extract_features, preprocess_audio, load_model, predict_emotion

class AudioFeatureExtractorApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Feature Extraction")

        self.model = load_model("model.h5")

        self.create_widgets()

    def create_widgets(self):
        self.browse_button = tk.Button(self.master, text="Wybierz plik", command=self.browse_file)
        self.browse_button.pack(pady=20)

        self.result_label = tk.Label(self.master, text="Output: ")
        self.result_label.pack()

    def process_audio_file(self, filepath):
        audio_data, sample_rate = preprocess_audio(filepath)
        features = extract_features(audio_data, sample_rate)
        predicted_emotion = predict_emotion(self.model, features)

        return predicted_emotion

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if filepath:
            result = self.process_audio_file(filepath)
            self.result_label.config(text=f"Output: {result}")

def main():
    root = tk.Tk()
    app = AudioFeatureExtractorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
