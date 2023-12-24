import os
import random
import librosa
import numpy as np

class NoiseAdder:
    def __init__(self, noise_directory):
        """
        noise_directory: Directory containing noise audio files.
        """
        self.noise_directory = noise_directory
        

    def select_noise_file(self):
        # Select a random noise file from the directory
        noise_files = [f for f in os.listdir(self.noise_directory) if f.endswith('.wav')]
        selected_noise_file = random.choice(noise_files)
        return os.path.join(self.noise_directory, selected_noise_file)

    def match_length_and_add_noise(self, signal, sr):
        # Load a random noise file
        noise_file = self.select_noise_file()
        noise, _ = librosa.load(noise_file, sr=sr)

        # Resize the noise to match the signal length
        if len(noise) > len(signal):
            noise = noise[:len(signal)]
        else:
            noise = np.pad(noise, (0, len(signal) - len(noise)), 'constant', constant_values=0)

        # Add the noise to the signal
        noised_signal = signal + noise

        # Normalize the noised signal
        noised_signal = librosa.util.normalize(noised_signal)
        
        return noised_signal
        