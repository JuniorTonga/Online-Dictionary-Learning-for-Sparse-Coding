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

