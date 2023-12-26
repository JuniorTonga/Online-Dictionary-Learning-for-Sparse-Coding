import pandas as pd
import matplotlib.pyplot as plt

class TimeSeriesPlotter:
    def __init__(self, dataframe):
        """
        dataframe: The pandas DataFrame containing time series data.
        """
        self.dataframe = dataframe

    def plot_indices(self, indices):
        """
        indices: A list of indices to plot from the DataFrame.
        """
        plt.figure(figsize=(12, 5))
        for idx in indices:
            if idx in range(self.dataframe.shape[0]):
                plt.plot(self.dataframe[idx,], label=f'Row {idx}')
            else:
                print(f"Index {idx} not found in the DataFrame.")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Plot')
        plt.legend()
        plt.show()


def plot_signals_and_reconstructions(signals, reconstructions, titles, title):
    """
    Plot original signals and their reconstructions in a 2x2 subplot.

    :param signals: List of 4 original signals.
    :param reconstructions: List of 4 reconstructed signals.
    :param titles: List of 4 titles for the subplots.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()  

    for i in range(4):
        axs[i].plot(signals[i], label='Original Signal')
        axs[i].plot(reconstructions[i], label='Reconstructed Signal', linestyle='--')
        axs[i].set_title(titles[i])
        axs[i].legend()
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.suptitle(title, fontsize=16) 
    plt.title(title)
    plt.show()



