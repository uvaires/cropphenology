import geopandas as gpd
import pandas as pd
import glob
import os
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt





def phenocam_processor(phenocam_data, interval=1):
    """
    Process the PhenoCam data

    Args:
        phenoCam_data (str): Path to the PhenoCam data CSV file.
        interval (int): Interval in days for filtering the data, change the interval to obtain the data in other intervals.

    Returns:
        pd.DataFrame: Filtered PhenoCam data.

    """
    # processing the CSV file
    phenocam_df = pd.read_csv(phenocam_data)
    # Convert the date column to datetime format
    phenocam_df['date'] = pd.to_datetime(phenocam_df['date'])
    # Sort the DataFrame by date
    phenocam_df = phenocam_df.sort_values(by='date')
    # Initialize a list to store filtered rows
    filtered_rows = []
    # Initialize variables to keep track of the previous date and the index
    prev_date = None
    index = 0

    # Iterate through each row in the DataFrame
    for _, row in phenocam_df.iterrows():
        # Check if it's the first row or if the current date is within a three-day interval from the previous date
        if prev_date is None or (row['date'] - prev_date).days >= interval or pd.isna(row['date']):
            # Add the row to the filtered rows list
            filtered_rows.append(row)
            # Update the previous date and index
            prev_date = row['date']
            index += 1

    # Create a new DataFrame containing only the filtered rows
    filtered_phenocam = pd.DataFrame(filtered_rows)

    return filtered_phenocam



def processing_hls_images(base_dir: str, station_name: str, polygon_path: str, year:str):
    """
    Process HLS images: remove clouds, select bands of interest

    Args:

        base_dir (str): Output directory where processed images are stored.
        station_name (str): Name of the station.

    """
    # reading the polygon data
    polygon_data = gpd.read_file(polygon_path)
    # Filter polygon data based on station name
    selected_polygon = polygon_data[polygon_data['station'] == station_name]
    # Find all HLS images in the directory
    img_dir = glob.glob(os.path.join(base_dir, '**', station_name, year, 'smoothed_evi', '*.tif'), recursive=True)

    # Reproject the selected polygon to the UTM projection using the CRS of the first image
    with rasterio.open(img_dir[0]) as src:
        target_crs = src.crs
        selected_polygon_utm = selected_polygon.to_crs(target_crs)

    # Iterate over each image and calculate mean EVI
    mean_evi_values = []
    date_doy = []
    for img in img_dir:
        # get the date from the image
        date = os.path.basename(img).split('_')[0]
        # convert the date 'YYYYMMDD' to Day of the Year (DOY)
        doy = pd.to_datetime(date, format='%Y%m%d').timetuple().tm_yday
        date_doy.append(doy)
        with rasterio.open(img) as src:
            # Mask the image with the reprojected polygon and calculate mean EVI
            masked_image, _ = mask(src, selected_polygon_utm.geometry, crop=True)
            # Calculate mean EVI, ignoring NaN values
            mean_value = np.nanmean(masked_image)
            if not np.isnan(mean_value):  # Check if the mean value is not NaN
                mean_evi_values.append(mean_value)
            else:
                mean_evi_values.append(np.nan)  # Insert NaN if mean value is NaN

    return mean_evi_values, date_doy


def process_hls_original(base_dir: str, station_name: str, polygon_path: str, year: str):
    # reading the polygon data
    polygon_data = gpd.read_file(polygon_path)
    # Filter polygon data based on station name
    selected_polygon = polygon_data[polygon_data['station'] == station_name]
    # Find all HLS images in the directory
    img_dir = glob.glob(os.path.join(base_dir, '**', station_name, year, 'spectral_index','**', '*.tif'),
                        recursive=True)

    # Reproject the selected polygon to the UTM projection using the CRS of the first image
    with rasterio.open(img_dir[0]) as src:
        target_crs = src.crs
        selected_polygon_utm = selected_polygon.to_crs(target_crs)

    # Iterate over each image and calculate mean EVI
    date_doy = []
    original_evi_values = []
    for img in img_dir:
        # get the date from the image
        date = os.path.basename(img).split('_')[0]
        # convert the date 'YYYYMMDD' to Day of the Year (DOY)
        doy = pd.to_datetime(date, format='%Y%m%d').timetuple().tm_yday
        date_doy.append(doy)
        with rasterio.open(img) as src:
            # Mask the image with the reprojected polygon and calculate mean EVI
            masked_image, _ = mask(src, selected_polygon_utm.geometry, crop=True)
            # Calculate mean EVI, ignoring NaN values
            mean_value = np.nanmean(masked_image)
            if not np.isnan(mean_value):  # Check if the mean value is not NaN
                original_evi_values.append(mean_value)
            else:
                original_evi_values.append(np.nan)  # Insert NaN if mean value is NaN

    return original_evi_values, date_doy


# Plotting the GCC Phenocam Mean and EVI HLS Mean
def plot_phenology(filtered_phenocam, mean_evi_values, date_doy_mean, original_evi_values, date_doy,
                                 station_name, base_dir, year):
    """
    Plot the GCC Phenocam Mean and EVI HLS Mean.

    Args:
        filtered_phenocam (pd.DataFrame): Filtered Phenocam data.
        mean_evi_values (List[float]): List of mean EVI values.
        station_name (str): Name of the station.

    """
    # Extracting dates from the "doy" column
    dates = filtered_phenocam['doy']

    # Creating the figure and axes with a larger figure size
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting "gcc_mean" from filtered_phenocam on the primary y-axis with larger markers
    ax1.plot(dates, filtered_phenocam['gcc_90'], color='blue', marker='o',linestyle='None', markersize=5, label='GCC Phenocam Mean')
    ax1.set_xlabel('DOY', fontname='Times New Roman', fontsize=12)
    ax1.set_ylabel('GCC Mean', color='black', fontname='Times New Roman', fontsize=12)

    # Adding a title to the plot
    plt.title('Comparison of GCC Phenocam Mean and EVI HLS Mean', fontname='Times New Roman', fontsize=14)

    # Adding grid lines to both axes
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # Creating a secondary y-axis
    ax2 = ax1.twinx()
    # Plotting mean_evi_values on the secondary y-axis with larger markers
    ax2.plot(date_doy_mean, mean_evi_values, color='green', marker='o', markersize=5, label='EVI HLS Synthetic')
    ax2.plot(date_doy, original_evi_values, color='red', marker='^', linestyle='None', markersize=7, label='EVI HLS Original')

    ax2.set_ylabel('EVI Mean', color='black', fontname='Times New Roman', fontsize=12)

    # Adjusting tick parameters for both axes
    ax1.tick_params(axis='x', rotation=0, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)

    # Show legend for both axes with larger font size
    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    # Adjusting the layout to prevent overlap of labels
    plt.tight_layout()
    plot_path = os.path.join(base_dir, 'data_processed', station_name, 'plots')
    os.makedirs(plot_path, exist_ok=True)

    save_plot = os.path.join(plot_path, f'{station_name}_{year}.png')
    # Save the plot to the specified output path
    plt.savefig(save_plot, dpi=300)
    # Show the plot
    plt.show()







