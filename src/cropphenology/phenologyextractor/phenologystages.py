import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
import matplotlib.dates as mdates

# Set Times New Roman as the font
plt.rcParams['font.family'] = 'Times New Roman'


def extract_phenological_dates(data_path: str, base_dir: str, station_name: str, vegetation_index='evi', year='2023',
                               interval=1):
    """
    Extract the phenological stages from the vegetation index time series
    :param data_path: time series data
    :param base_dir: output directory
    :param station_name: PhenoCam station name
    :param vegetation_index: column name of the vegetation index
    :param year: 2023
    :param interval: interval dates in the time series data
    :return: phenological stages
    """
    # Read the data
    if isinstance(data_path, pd.DataFrame):
        # If the data is already a DataFrame, just return it
        df = data_path
    elif isinstance(data_path, str) and os.path.exists(data_path):
        # If the data is a valid file path, read it using pandas
        df = pd.read_csv(data_path)

    # Convert the 'date' column to datetime format for easier time-based operations
    df['date'] = pd.to_datetime(df['date'])

    # Set the variable for the vegetation index (e.g., GCC, NDVI)
    vegetation_index = vegetation_index

    # Define the interval for resampling (1 day in this case)
    interval = interval

    # Set the dataframe index to the 'date' column for easier time-based operations
    df.index = df.date

    # Resample the vegetation index time series to the specified interval (daily)
    # Take the maximum value within each resampling window and interpolate missing values
    vi_time_serie = (df.loc[:, vegetation_index].resample(f'{interval}d').max()
                     .interpolate(method="time", limit_direction='both'))

    # Create the xdata (time in days) and ydata (the vegetation index values) for curve fitting
    xdata = np.array(range(vi_time_serie.shape[0]))
    ydata = np.array(vi_time_serie)

    # Initial guess for the parameters of the asymmetric double sigmoid model
    p0 = [0.2, 0.6, 0.05, 50 / interval, 0.05, 130 / interval]

    # Set bounds for the parameters to constrain the optimization process
    bounds = ([0.0, 0.2, -np.inf, 0, 0, 0], [0.5, 0.8, np.inf, xdata.shape[0], 0.4, xdata.shape[0]])

    # Perform curve fitting to the data using the asymmetric double sigmoid model
    # This will estimate the parameters that best fit the time series data
    popt, pcov = opt.curve_fit(_asymmetric_dbl_sigmoid_model,
                               xdata=xdata,
                               ydata=ydata,
                               p0=p0,
                               bounds=bounds,
                               method='trf',
                               maxfev=10000)

    # If the covariance matrix could not be estimated, raise an error
    if np.any(np.isinf(pcov)):
        raise RuntimeError("Covariance of the parameters could not be estimated")

    # Extract the fitted parameters (Vb, Va, p, Di, q, Dd) from the optimization result
    Vb, Va, p, Di, q, Dd = popt

    # Use the fitted parameters to compute the fitted vegetation index curve
    vi_fitted = _asymmetric_dbl_sigmoid_model(xdata, *popt)

    # Calculate the phenological transition dates based on the fitted model parameters
    D1 = Di + np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D2 = Di - np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D3 = Dd + np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D4 = Dd - np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)

    # Plot the raw vegetation index time series and the fitted curve
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the raw vegetation index data (gray dots)
    ax.plot(df.index, df[vegetation_index], marker='.', lw=0, label=f'Raw {vegetation_index}', color='gray')

    # Plot the fitted curve (blue line)
    ax.plot(vi_time_serie.index, vi_fitted, label=f'Fitted {vegetation_index}', color='blue')

    # Set plot limits and labels
    ax.set_ylim(0, 1)
    ax.set_ylabel(f'{vegetation_index}', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_title(f'Phenological Stages', fontsize=16, fontweight='bold')
    ax.grid(True)

    # Define the colors and labels for each phenological stage
    colors = ['green', 'blue', 'black', 'purple', 'orange', 'red']
    labels = ['Greenup', 'MidGreenup', 'Maturity', 'Senescence', 'MidGreendown', 'Dormancy']

    # Add vertical lines at the calculated phenological transition dates
    for i, d in enumerate([D1, Di, D2, D3, Dd, D4]):
        ax.axvline(vi_time_serie.index[int(round(d))], 0, vi_fitted[int(round(d))], color=colors[i],
                   label=f'{labels[i]}: {str(vi_time_serie.index[int(round(d))].date())}', ls='--', lw=1.5)

    # Add a legend to the plot
    ax.legend(frameon=True)

    # Create directories for saving the plot if they don't exist
    plot_path = os.path.join(base_dir, 'data_processed', station_name, year, 'plots')
    os.makedirs(plot_path, exist_ok=True)

    # Save the plot as a PNG file
    file_path = os.path.join(plot_path, f'phenology_stages{station_name}_{year}.png')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Rotate the x-axis labels (date labels)
    for label in ax.get_xticklabels():
        label.set_rotation(0)  # Set rotation to 0 degrees for horizontal text

    plt.savefig(file_path, dpi=300)
    plt.show()

    # Save the calculated phenological dates (D1, Di, D2, D3, Dd, D4) as a dictionary
    phenological_dates = {
        'D1': vi_time_serie.index[int(round(D1))].strftime('%Y%m%d'),
        'Di': vi_time_serie.index[int(round(Di))].strftime('%Y%m%d'),
        'D2': vi_time_serie.index[int(round(D2))].strftime('%Y%m%d'),
        'D3': vi_time_serie.index[int(round(D3))].strftime('%Y%m%d'),
        'Dd': vi_time_serie.index[int(round(Dd))].strftime('%Y%m%d'),
        'D4': vi_time_serie.index[int(round(D4))].strftime('%Y%m%d')
    }

    # Convert the phenological dates dictionary into a DataFrame and save it as a CSV file
    phenological_dates_df = pd.DataFrame(phenological_dates, index=[0])

    return phenological_dates_df


# Private function to define the asymmetric double sigmoid model
def _asymmetric_dbl_sigmoid_model(t: np.array, Vb: float, Va: float, p: float, Di: float, q: float, Dd: float):
    """A double logistic model, as in Zhong et al 2016"""
    return Vb + 0.5 * Va * (np.tanh(p * (t - Di)) - np.tanh(q * (t - Dd)))
