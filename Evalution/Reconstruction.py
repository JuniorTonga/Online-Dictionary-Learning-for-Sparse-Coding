from sklearn.decomposition import SparseCoder
import numpy as np

class Evaluation:
    def __init__(self, dictionary):
        """
        dictionary: Learned dictionary (atoms).
        """
        self.dictionary = dictionary

    def reconstruct_signal(self, input_signal):
        """
        input_signal: The input signal to be reconstructed.
        """
        # Initialize the SparseCoder with the learned dictionary
        coder = SparseCoder(dictionary=self.dictionary, transform_algorithm='lasso_lars', transform_alpha=0.1)

        # Flatten the input signal if it's not already flat
        input_signal_flat = input_signal.flatten() if input_signal.ndim > 1 else input_signal

        # Use the SparseCoder to find the sparse representation
        sparse_representation = coder.transform([input_signal_flat])

        # Reconstruct the signal from the sparse representation
        reconstructed_signal = np.dot(sparse_representation, self.dictionary)

        return reconstructed_signal[0]

