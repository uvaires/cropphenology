import pandas as pd
from scipy.signal import savgol_filter


def normalize_gcc(phenocam_path: str, gcc='gcc_mean', window_length=11, polyorder=3):
    # Load GCC data from the CSV file
    gcc_data = pd.read_csv(phenocam_path)

    # Extract GCC values and DOY (Day of Year)
    gcc_values = gcc_data[gcc]

    # Apply Min-Max normalization
    min_gcc = gcc_values.min()
    max_gcc = gcc_values.max()
    normalized_gcc = (gcc_values - min_gcc) / (max_gcc - min_gcc)

    # Apply Savitzky-Golay filter to the normalized GCC data
    smoothed_gcc = savgol_filter(normalized_gcc, window_length, polyorder)

    # Add the normalized and smoothed values to the dataframe
    gcc_data['GCC'] = smoothed_gcc

    return gcc_data
