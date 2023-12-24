import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display

class AudioVisualization:
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths

    def display_waveform(self, audio_path):
        audio, sr = librosa.load(audio_path)
        plt.figure(figsize=(12, 4))
        plt.plot(audio)
        plt.title(f"Waveform of {audio_path}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    def play_audio(self, audio_path):
        return Audio(audio_path)

    def show(self):
        for audio_path in self.audio_paths:
            self.display_waveform(audio_path)
            display(self.play_audio(audio_path))


class AudioVisualizer:
    def __init__(self, audio_data, sample_rate=22050):
        """
        audio_data: NumPy array containing the audio data.
        sample_rate: Sample rate of the audio data. Default is 22050 Hz.
        """
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def visualize(self):
        """
        Visualize the audio waveform.
        """
        plt.figure(figsize=(12, 4))
        plt.plot(self.audio_data)
        plt.title("Audio Waveform")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    def play(self):
        """
        Play the audio data.
        """
        display(Audio(data=self.audio_data, rate=self.sample_rate))

    def show(self):
        """
        Show the visualization and play the audio.
        """
        self.visualize()
        self.play()


