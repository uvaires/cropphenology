import numpy as np
from datetime import timedelta
from collections import Counter
from geogapfiller import utils
from geogapfiller import gapfiller



def run_method(filler:gapfiller.Filler, img_list: list, base_dates: list, interval: int)->tuple:
    """
      Run the method to predict the images
      :param filler: Instance of a Filler subclass from gapfiller (e.g., MedianFiller, PolynomialFiller)
      :param img_list: List of rasters (images) to be processed
      :param base_dates: List of base dates corresponding to the rasters
      :param interval: Time interval in days for prediction
      :return: Predicted raster and date range
      """

    dates_counter = Counter(base_dates)
    start_date = min(base_dates)
    end_date = max(base_dates)

    # Calculate the date range based on the specified interval
    date_range = [start_date + timedelta(days=i) for i in range(0, (end_date - start_date).days + 1, interval)]

    # Stack the list of rasters
    stack_rasters = utils._stack_raster(img_list)

    # Create an array for the predicted images with the same dimensions
    original_arr = np.full((len(date_range), stack_rasters.shape[1], stack_rasters.shape[2]), np.nan)

    # Fill in the original array with the available raster data
    for i, date in enumerate(base_dates):
        if start_date <= date <= end_date:
            try:
                date_index = date_range.index(date)

                if dates_counter[date] > 1:
                    existing_values = original_arr[date_index, :, :]
                    new_values = stack_rasters[i, :, :]
                    combined_values = np.nanmean([existing_values, new_values], axis=0)
                else:
                    combined_values = stack_rasters[i, :, :]

                original_arr[date_index, :, :] = combined_values
            except ValueError:
                pass  # Date is outside the specified interval

    # predicted_raster = filler.fi(original_arr, date_range)
    predicted_raster = gapfiller.run_method(filler, original_arr, date_range)

    return predicted_raster, date_range