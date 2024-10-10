import os
import numpy as np
import rasterio
import glob
from datetime import datetime

def _stack_raster(img_list):
    raster_layers = []
    for img in img_list:
        with rasterio.open(img) as src:
            raster_layer = src.read(1)
            raster_layers.append(raster_layer)

    # stack the EVI layers
    stacked_raster = np.stack(raster_layers, axis=0)
    stacked_raster = np.squeeze(stacked_raster)

    return stacked_raster

def _img_metadata(img_list):
    with rasterio.open(img_list[0]) as src:  # Use the first image in the list
        img_profile = src.profile  # Extract the metadata profile

    return img_profile

# Convert dates to string
def _convert_dates(dates):
    base_dates = [date.strftime("%Y%m%d") for date in dates]
    return base_dates

# Use this function to create a list of images and dates in the format of datetime
def img_pattern(img_path, pattern: str)-> tuple:
    """ Function to get a list of images and dates in the format of datetime
    Args:
        img_path (str): Path to the images
        pattern (str): Pattern to search for the images in the folder
    Returns: list of images and dates in the format of datetime"""

    # Get a list of all raster files matching the pattern
    img_list = glob.glob(os.path.join(img_path, '**', f"*{pattern}.tif"), recursive=True)
    # Create an empty list to store the dates
    img_dates = []
    for img in img_list:
        img_date = os.path.basename(img).split("_")[0]  # Extract the date part from the filename
        convert_date = datetime.strptime(img_date, '%Y%m%d')
        img_dates.append(convert_date)  # Store both the file and the formatted date

    return img_list, img_dates

def export_raster(outputdir:list, img_list:list, raster_filled:tuple, img_dates:list, station_name:str)->None:
    """
 Export the filled EVI images as GeoTIFF files.
    :param outputdir: Output directory (as a string)
    :param img_list: List of image files
    :param raster_filled: 3D array (n_layers, height, width) of the filled raster data
    :param img_dates: List of dates corresponding to the images (datetime format)
    :param method: Method used to fill the gaps (e.g., 'harmonic', 'lightgbm')
    :param pattern: Pattern (e.g., band name like 'B02', 'NIR') to be used in the filename
    :return: None
    """
    # Open the first image to get the profile
    img_profile = _img_metadata(img_list)
    # convert dates into string
    img_dates = _convert_dates(img_dates)
    year = img_dates[0][:4]
    # Output the filled images
    outpupath = os.path.join(outputdir,'data_processed', station_name, year, 'predicted_img')

    for layer_data, current_dates in zip(raster_filled, img_dates):
             # Construct the output file path for the current layer
            dir_output_data = os.path.join(outpupath)
            os.makedirs(dir_output_data, exist_ok=True)
            output_filename = f"{current_dates}_evi.tif"
            output_filepath = os.path.join(dir_output_data, output_filename)

            # Write the filled EVI image to a GeoTIFF file
            with rasterio.open(output_filepath, 'w', **img_profile) as dst:
                dst.write(layer_data, 1)
