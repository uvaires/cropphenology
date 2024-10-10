import numpy as np
from scipy.signal import savgol_filter
import rasterio
import os
import glob
from datetime import datetime


# Function to apply the Savitzky-Golay filter to the EVI predicted images
def apply_sg_filter(base_dir: str, station_name: str, window_length=10, poly_order=3):
    """
    Function to apply the Savitzky-Golay filter to the EVI predicted images
    :param base_dir:    base directory where the EVI predicted images are stored
    :param station_name:    station name
    :param window_length:   window length for the Savitzky-Golay filter
    :param poly_order:  polynomial order for the Savitzky-Golay filter
    :return: none
    """

    # Read the EVI predicted images
    evi_pred = glob.glob(os.path.join(base_dir, 'data_processed', station_name, '**', 'predicted_img', '*.tif'),
                         recursive=True)
    # Get the EVI predicted profile and layers
    evi_layers, img_profile = _load_evi_layers(evi_pred)

    # Stack the EVI layers
    evi_pred_stack = np.stack(evi_layers, axis=0)

    # Extract the base dates from images
    base_dates, tile_id, years = _extract_base_dates(evi_pred)

    # Apply the Savitzky-Golay filter
    evi_smooth = _apply_savgolay(evi_pred_stack, window_length, poly_order)

    # Export the smoothed EVI images
    _export_img(evi_smooth, base_dates, base_dir, img_profile, station_name, years)


### Privite functions ##
# Extract the band profile
def _load_evi_layers(evi_img):
    evi_layers = []
    band_profile = None

    # Loop through the EVI images and load them
    for evi_path in evi_img:
        with rasterio.open(evi_path) as src:
            evi_layer = src.read(1)
            band_profile = src.profile
            evi_layers.append(evi_layer)

    return evi_layers, band_profile


# Extract information from the image file name such as date and product
def _extract_base_dates(evi_img):
    base_dates = []
    years = []
    tile_ids = []

    for dates in evi_img:
        date_type = os.path.basename(dates).split('_')[0]
        convert_date = datetime.strptime(date_type, '%Y%m%d')
        year_str = str(convert_date.year)  # Convert year to string

        tile_id = os.path.basename(dates).split('_')[1]

        base_dates.append(date_type)
        years.append(year_str)
        tile_ids.append(tile_id)

    return base_dates, tile_ids, years


# Export the smoothed EVI images

def _export_img(evi_smooth, base_dates, base_dir, img_profile, station_name, years):
    for layer_idx, (layer_data, current_date, year) in enumerate(zip(evi_smooth, base_dates, years)):
        # Construct the output directory path for the current layer
        dir_output_date = os.path.join(base_dir, 'data_processed', station_name, year, 'smoothed_evi')
        os.makedirs(dir_output_date, exist_ok=True)

        # Construct the output file path for the current layer
        output_filepath = os.path.join(dir_output_date, f"{current_date}_evi.tif")

        # Update the image profile
        img_profile['nodata'] = np.nan
        img_profile['compress'] = 'lzw'

        # Write the data to the file
        with rasterio.open(output_filepath, 'w', **img_profile) as dst:
            dst.write(layer_data, 1)


# Function to smooth the EVI images
def _apply_savgolay(evi_raster_filled, window_length, poly_order):
    evi_savitzky_golay = np.empty_like(evi_raster_filled)

    for i in range(evi_raster_filled.shape[1]):
        for j in range(evi_raster_filled.shape[2]):
            evi_values = evi_raster_filled[:, i, j]

            if np.isnan(evi_values).all():
                continue

            # Ignore NaN values and apply Savitzky-Golay filter
            non_nan_indices = ~np.isnan(evi_values)
            smoothed_evi = np.full_like(evi_values, np.nan)
            smoothed_evi[non_nan_indices] = savgol_filter(evi_values[non_nan_indices], window_length, poly_order,
                                                          mode='wrap')

            evi_savitzky_golay[:, i, j] = smoothed_evi

    return evi_savitzky_golay
