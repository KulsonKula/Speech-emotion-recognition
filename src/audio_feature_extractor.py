import tkinter as tk
from tkinter import filedialog
from utils import audio_preprocessing
from utils import model
import sys
import os
import sounddevice
from scipy.io.wavfile import write

backgrd='gray16'
foregrd='white'
bckgrd2='gray25'


sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


class AudioFeatureExtractorApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Feature Extraction")

        self.model = model.load_model("model.h5")

        self.create_widgets()

    def create_widgets(self):

        self.frame = tk.Frame(self.master, width=250, height=600, background=bckgrd2)
        self.frame.place(x=0,y=0)

        self.frameCentral = tk.Frame(self.master, width=450, height=450, background=bckgrd2)
        self.frameCentral.place(x=300,y=100)

        self.frameOutput = tk.Frame(self.frameCentral, width=150, height=50, background='gray40')
        self.frameOutput.place(x=150,y=230)

        self.label = tk.Label(self.master, text="Speech emotion recognition app", font=['Times new roman',18,'bold'], foreground='SpringGreen3', background=backgrd) 
        self.label.place(x=260,y=10)

        self.labelLeft = tk.Label(self.frame, text="Choose the voice input for \n emotion recognition:", font=['Times new roman',12,'bold'], foreground=foregrd, background=bckgrd2) 
        self.labelLeft.place(x=30,y=60)
        
        self.browse_button = tk.Button(self.frame, text="Choose file", width=20, font=['Times new roman',12,'bold'], command=self.browse_file, background='SpringGreen3', foreground=foregrd, activebackground='SpringGreen4')
        self.browse_button.place(x=30,y=150)

        self.record_button = tk.Button(self.frame, text="Record your speech",width=20,font=['Times new roman',12,'bold'], command=self.display_textRecording, background='SpringGreen3',foreground=foregrd, activebackground='SpringGreen4')
        self.record_button.place(x=30,y=200)

        self.result_label = tk.Label(self.frameCentral, text="Detected emotion: ", font=['Times new roman',14,'bold'], background=bckgrd2,foreground=foregrd)
        self.result_label.place(x=150,y=200)

        self.labelCentral = tk.Label(self.frameCentral, text="The results of emotion recognition", font=['Times new roman',12,'bold'], foreground=foregrd, background=bckgrd2) 
        self.labelCentral.place(x=100,y=10)

    



    def process_audio_file(self, filepath):
        audio_data, sample_rate = audio_preprocessing.preprocess_audio(filepath)
        features = audio_preprocessing.extract_features(audio_data, sample_rate)
        predicted_emotion = model.predict_emotion(self.model, features)

        return predicted_emotion
        

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if filepath:
            result = self.process_audio_file(filepath)
            self.labelResult=tk.Label(self.frameOutput,text=f"{result}",font=['Times new roman',14,'bold'],background='gray40',foreground=foregrd)
            self.labelResult.pack()
            self.labelResult.place(x=10, y=10)

    def display_textRecording(self):
        self.recording_start = tk.Label(self.frame, text="Start recording     ", background=bckgrd2,font=['Times new roman',15,'bold'],foreground=foregrd)
        self.recording_start.place(x=30,y=250)
        self.frame.after(10, self.record_speech)

    def record_speech(self):

        sampe_rate = 22050
        second = 3
        filename="ERapp.wav"
        speech_recording = sounddevice.rec(int(second * sampe_rate), samplerate=sampe_rate, channels=2)
        sounddevice.wait()
        write(filename, sampe_rate, speech_recording)

        self.recording_stop = tk.Label(self.frame, text="Stopped recording", background=bckgrd2,font=['Times new roman',15,'bold'],foreground=foregrd)
        self.recording_stop.place(x=30,y=250)

        current_directory = os.path.dirname(os.curdir)
        filepath = os.path.join(current_directory, filename)
        result = self.process_audio_file(filepath)

        self.labelResult=tk.Label(self.frameOutput,text=f"{result}",font=['Times new roman',14,'bold'],background='gray40',foreground=foregrd)
        self.labelResult.place(x=10, y=10)
        os.remove(filepath)







def main():
    root = tk.Tk()
    root.geometry("800x600")
    root.config(background=backgrd)
    app = AudioFeatureExtractorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
