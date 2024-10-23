import os
import rasterio
import numpy as np
from decimal import Decimal
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from geogapfiller import gapfiller


def filling_techn(base_dir, img_list, img_dates, methods, img_validator, station_name, grid_distance=200,
                  offset=100) -> tuple:
    """
    Validate the filling gaps techniques
    :param base_dir: base directory to export the results
    :param img_list: list of images
    :param img_dates: list of dates of the images
    :param methods: list of methods to fill the gaps
    :param img_validator: list of images with 100% of the data
    :param station_name: PhenoCam station name
    :param grid_distance: define the distance between the grid points
    :param offset: define offset distance from the border
    :return: tuple with the observed values and the filled values for each method
     """

    # Read original images
    evi_original = img_list
    # Extract the indices of the images that match the selected dates
    selected_indices = _extract_date_indices(img_list, img_validator)

    # Generate regular grid points as a GeoDataFrame
    points_gdf = _generate_regular_grid(img_list[1], grid_distance, offset, base_dir, station_name, img_validator)

    # Stack observed EVI images
    evi_obs = _stack_raster_layers(img_list)
    # Stack the images and replace the original values with np.nan for selected indices
    evi_obs_with_nans = _stack_and_replace_images(img_list, selected_indices)

    # Create the list of functions from geogapfiller package
    technique_funcs = [gapfiller.get_method(method) for method in methods]

    # Dictionaries to hold the filled data for each method
    filled_nan = {}

    # Run the techniques for both NaN-filled data
    for func, method_name in zip(technique_funcs, methods):
        # Apply the gap-filling technique to the original data
        raster_filled_original = gapfiller.run_method(func, evi_obs_with_nans, img_dates)
        evi_technique_values = _extract_evi_values(raster_filled_original, points_gdf, selected_indices,
                                                   evi_original)
        filled_nan[method_name] = evi_technique_values

    # Extract the pixel of the original rasters
    evi_obs_values = _extract_evi_values(evi_obs, points_gdf, selected_indices, evi_original)

    return evi_obs_values, filled_nan


def plot_techniques_val(img_dates: list, evi_obs_values: list, evi_median_values: list, evi_poly_values: list,
                        evi_harmonic_values: list, evi_lightgbm_values: list, methods: list, base_dir: str,
                        station_name: str) -> None:
    """
    Plot the observed vs predicted values for each method
    :param img_dates: three dates of images with 100% of the data
    :param evi_obs_values: Original values of EVI
    :param evi_median_values: Predicted values of EVI using median method
    :param evi_poly_values: Predicted values of EVI using polynomial method
    :param evi_harmonic_values: Predicted values of EVI using harmonic method
    :param evi_lightgbm_values: Predicted values of EVI using lightgbm method
    :param methods: List of methods
    :param base_dir: used to save the plot and metrics
    :param station_name: PhenoCam station name

    :return: scaterplot of observed vs predicted values
    """
    year = img_dates[0].split('-')[0]

    # Create a dictionary with the observed and predicted values for each date
    img_data = {}

    for date, obs, median, poly, harmonic, lightgbm in zip(img_dates, evi_obs_values, evi_median_values,
                                                           evi_poly_values,
                                                           evi_harmonic_values,
                                                           evi_lightgbm_values):
        img_data[date] = {
            "Observed": obs,
            methods[0]: median,
            methods[1]: poly,
            methods[2]: harmonic,
            methods[3]: lightgbm,
        }

    # Initialize a subplot with 3 columns and number of rows as the number of methods
    fig, axes = plt.subplots(len(methods), 3, figsize=(18, 20))

    metrics_data = []

    for j, method in enumerate(methods):
        for i, (date_key, date_info) in enumerate(img_data.items()):
            observed_values = date_info["Observed"]
            predicted_values = date_info[method]

            rmse, mae, r2 = _calculate_metrics(observed_values, predicted_values)
            metrics_data.append({
                "Date": date_key,
                "Method": method,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            })

            # Calculate subplot index (1-indexed)
            subplot_index = j * len(img_data) + i + 1

            # Plot observed vs predicted values on the current subplot
            axes[j, i].scatter(observed_values, predicted_values, color='skyblue', edgecolors='black', alpha=0.7)

            # Add the 1:1 line
            min_val = min(min(observed_values), min(predicted_values))
            max_val = max(max(observed_values), max(predicted_values))
            axes[j, i].plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', linewidth=2)

            axes[j, i].set_xlabel('Observed', fontsize=12)
            axes[j, i].set_ylabel('Predicted', fontsize=12)
            axes[j, i].set_title(f'{method} - {date_key}', fontsize=14)

            rmse = Decimal(float(rmse)).quantize(Decimal('0.000'))
            mae = Decimal(float(mae)).quantize(Decimal('0.000'))
            r2 = Decimal(float(r2)).quantize(Decimal('0.00'))

            axes[j, i].text(0.01, 0.9, f'RMSE: {rmse}\nMAE: {mae}\nR2: {r2}',
                            transform=axes[j, i].transAxes,
                            verticalalignment='top', horizontalalignment='left', color='black',
                            fontsize=10)

            # Add grid lines
            axes[j, i].grid(True, linestyle='--', alpha=0.5)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Create directories for saving plot and metrics
    save_plot = os.path.join(base_dir, 'data_processed', station_name, year, 'plots')
    os.makedirs(save_plot, exist_ok=True)
    plotout_dir = os.path.join(save_plot, 'evi_filled_pred_obs.png')

    # Save the plot as an image file
    plt.savefig(plotout_dir, bbox_inches='tight', dpi=300)
    # Show the plots
    plt.show()

    # Convert metrics data to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    save_metrics = os.path.join(base_dir, 'data_processed', station_name, year, 'validation_metrics')
    os.makedirs(save_metrics, exist_ok=True)

    # Export metrics DataFrame to Excel file
    excel_file_path = os.path.join(save_metrics, f'evi_{station_name}_metrics.xlsx')
    metrics_df.to_excel(excel_file_path, index=False)


# Sample function to calculate metrics (replace this with your own implementation)
def _calculate_metrics(observed_values, predicted_values):
    # Placeholder implementation, replace with actual implementation
    return 0, 0, 0


### Privite functions

def _generate_regular_grid(image_path, grid_step, offset, base_dir, station_name, img_validator):
    year = img_validator[0].split('-')[0]
    with rasterio.open(image_path) as src:
        # Create a GeoDataFrame with a regular grid of points
        x_min, y_max = src.bounds.left + offset, src.bounds.top - offset
        x_coords = np.arange(x_min, src.bounds.right, grid_step)
        y_coords = np.arange(y_max, src.bounds.bottom, -grid_step)

        grid_points = []
        for y in y_coords:
            for x in x_coords:
                grid_points.append(Point(x, y))

        gdf = gpd.GeoDataFrame(geometry=grid_points, crs=src.crs)

        # Transform the points to the CRS of the raster image
        gdf = gdf.to_crs(src.crs)
        output_path = os.path.join(base_dir, 'data_processed', station_name, year, 'validation_metrics')
        os.makedirs(output_path, exist_ok=True)
        save_grid = os.path.join(output_path, f'grid_{station_name}_points.shp')
        gdf.to_file(save_grid, src_crs=src.crs)

    return gdf


# Function to stack the raster
def _stack_raster_layers(image_paths):
    raster_layers = []

    for image_path in image_paths:
        with rasterio.open(image_path) as src:
            raster_layer = src.read(1)
            raster_layers.append(raster_layer)

    stacked_raster = np.stack(raster_layers, axis=0)

    return stacked_raster


# Function to replace the original values with np.nan for selected indices
def _stack_and_replace_images(image_paths, selected_indices):
    stack = []

    # Get the shape of the first image
    with rasterio.open(image_paths[0]) as src:
        rows, cols = src.shape

    # Initialize an array to store the stacked images
    all_images_stack = np.empty((len(image_paths), rows, cols))

    for i, image_path in enumerate(image_paths):
        try:
            with rasterio.open(image_path) as src:
                raster = src.read(1)

                # Resize the image if needed to match the shape of the first image
                if raster.shape != (rows, cols):
                    raster = rasterio.transform.resize(raster, (rows, cols))

                stack.append(raster)

            # Store the image in the array
            all_images_stack[i, :, :] = raster
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Stack all images
    all_images_stack_stacked = np.stack(stack, axis=0)

    # Replace the original values with np.nan for selected indices
    for i in selected_indices:
        all_images_stack[i, :, :] = np.full((rows, cols), np.nan)

    print(f"Total images in evi_img_original: {len(image_paths)}")
    print(f"Total images in stack: {all_images_stack_stacked.shape[0]}")

    return all_images_stack


# Function to extract the pixel indices corresponding to the points
def _points_to_pixel_indices(points_gdf, shape, evi_img):
    # Open the actual raster to get the transformation information
    with rasterio.open(evi_img) as src:
        transform = src.transform
        rows, cols = src.shape

    pixel_indices = []

    for point in points_gdf['geometry']:
        # Use the inverse transform to get pixel indices
        col, row = ~transform * (point.x, point.y)

        # Ensure indices are within bounds
        col = min(max(0, int(col)), cols - 1)
        row = min(max(0, int(row)), rows - 1)

        pixel_indices.append((row, col))

    return pixel_indices


def _extract_evi_values(evi_data, points_gdf, selected_indices, evi_img):
    extracted_values = []

    for idx in selected_indices:
        # Extract the EVI layer for the current index
        evi_layer = evi_data[idx, :, :]

        # Extract EVI values for each point in points_gdf
        evi_points = [evi_layer[rowcol] for rowcol in
                      _points_to_pixel_indices(points_gdf, evi_layer.shape, evi_img[idx])]

        extracted_values.append(evi_points)

    return extracted_values


# Function to calculate the metrics
def _calculate_metrics(observed, predicted):
    # Convert lists to NumPy arrays if they are not already
    observed = np.array(observed)
    predicted = np.array(predicted)

    # Ignore NaN values in observed and predicted arrays
    observed_non_nan = observed[~np.isnan(observed) & ~np.isnan(predicted)]
    predicted_non_nan = predicted[~np.isnan(observed) & ~np.isnan(predicted)]

    # Calculate mean squared error
    mse = np.nanmean((observed_non_nan - predicted_non_nan) ** 2)

    # Calculate root mean squared error
    rmse = np.sqrt(mse)

    # Calculate mean absolute error
    mae = np.nanmean(np.abs(observed_non_nan - predicted_non_nan))

    # Calculate Pearson correlation coefficient (ignoring NaN values)
    r = np.corrcoef(observed_non_nan, predicted_non_nan)[0, 1]
    r2 = r ** 2

    return rmse, mae, r2


def _extract_date_indices(evi_original, img_dates):
    # Extract the dates from the image names
    img_metadata = [os.path.basename(img).split('_')[0] for img in evi_original]
    # convert dates to datetime objects
    img_metadata_convert = [datetime.strptime(date, "%Y%m%d") for date in img_metadata]

    # List to store indices of matching dates
    selected_indices = []

    img_date_indice = img_dates.copy()

    # Iterate over the dates in img_metadata_convert and check for matches
    for i, date in enumerate(img_metadata_convert):
        # Convert the datetime object to a string in the format 'YYYY-MM-DD'
        date_str = date.strftime('%Y-%m-%d')
        # Check if the date string is in img_dates and if it's not already been matched
        if date_str in img_date_indice:
            # If a match is found, store the index
            selected_indices.append(i)
            # Remove the matched date from img_dates to avoid duplicate matches
            img_date_indice.remove(date_str)

    # Print the indices corresponding to the matching dates
    print("Indices of matching dates:")
    print(selected_indices)

    return selected_indices
