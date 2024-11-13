import pandas as pd
import glob
import os
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
from sklearn.exceptions import ConvergenceWarning
import geopandas as gpd
import joblib
from joblib import Parallel, delayed
from shapely.errors import ShapelyDeprecationWarning
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# Suppress Shapely Deprecation Warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# Suppress warnings related to model convergence and joblib's UserWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

setPYTHONWARNINGS="ignore::shapely.errors.ShapelyDeprecationWarning"

# Asymmetric double sigmoid model function
def _asymmetric_dbl_sigmoid_model(t, Vb, Va, p, Di, q, Dd):
    """A double logistic model, as in Zhong et al 2016"""
    return Vb + 0.5 * Va * (np.tanh(p * (t - Di)) - np.tanh(q * (t - Dd)))


# Convert Day of Year (DOY) to date
def _convert_doy_to_date(year, doy):
    start_date = f'{year}-01-01'
    start_date = pd.to_datetime(start_date)
    target_date = start_date + pd.to_timedelta(doy - 1, unit='d')
    date_str = target_date.strftime('%Y%m%d')
    return date_str


def _predict_dates(row, planting_model, emergence_model):
    features = np.array([[row['D1'], row['Di'], row['D2'], row['D3'], row['Dd'], row['D4']]])
    planting_date = planting_model.predict(features)[0]
    planting_date = round(planting_date)
    emergence_date = emergence_model.predict(features)[0]
    emergence_date = round(emergence_date)

    return planting_date, emergence_date


def _extract_evi_data(img_dir, selected_polygon_utm):
    mean_evi_values = []
    date_doy = []

    # Iterate over each image and calculate mean EVI
    for img in img_dir:
        # Get the date from the image
        date = os.path.basename(img).split('_')[0]
        # Convert the date 'YYYYMMDD' to Day of the Year (DOY)
        doy = pd.to_datetime(date, format='%Y%m%d').timetuple().tm_yday
        date_doy.append(doy)

        with rasterio.open(img) as src:
            # Mask the image with the reprojected polygon and calculate mean EVI
            masked_image, _ = mask(src, selected_polygon_utm.geometry, crop=True)
            # Calculate mean EVI, ignoring NaN values
            mean_value = np.nanmean(masked_image)
            mean_evi_values.append(mean_value if not np.isnan(mean_value) else np.nan)

    return mean_evi_values, date_doy


def _extract_projection(img_dir):
    with rasterio.open(img_dir[0]) as src:
        target_crs = src.crs
        return target_crs


def _verify_and_filter_field(polygon_data, field):

    # Filter polygon data based on the current field
    selected_polygon = polygon_data[polygon_data['field_coun'] == field]

    if selected_polygon.empty:
        return None, None, None

    # Get the crop type for the current field
    crop_type = selected_polygon['crop'].values[0]

    # Check if the crop type is not Corn or Soybeans
    if crop_type not in ['Corn', 'Soybeans']:
        print(f"Skipping field {field} with crop type {crop_type}")
        empty_result = {
            'field': field,
            'D1': '',
            'Di': '',
            'D2': '',
            'D3': '',
            'Dd': '',
            'D4': ''
        }
        return None, crop_type, empty_result

    return selected_polygon, crop_type, None


def _create_dataframe(mean_evi_values, date_doy, year, vegetation_index='mean_evi'):
    df = pd.DataFrame({
        'date': pd.to_datetime([_convert_doy_to_date(year, doy) for doy in date_doy]),
        vegetation_index: mean_evi_values
    })
    df = df.set_index('date')

    return df


def _plot_phenological_dates(df, vi_time_serie, vi_fitted, D1, Di, D2, D3, Dd, D4, vegetation_index='mean_evi'):
    # Plot the time series
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[vegetation_index], marker='.', lw=0, label='Raw VI')
    ax.plot(vi_time_serie.index, vi_fitted, label='Fitted VI')
    ax.set_ylim(0, 1)
    ax.set_ylabel(vegetation_index)

    colors = ['m', 'g', 'yellow', 'c', 'orange', 'b']
    labels = ['D1', 'Di', 'D2', 'D3', 'Dd', 'D4']
    for i, d in enumerate([D1, Di, D2, D3, Dd, D4]):
        ax.axvline(vi_time_serie.index[int(round(d))], 0, vi_fitted[int(round(d))], color=colors[i],
                   label=f'{labels[i]}: {vi_time_serie.index[int(round(d))].date()}', ls='--')

    ax.legend()
    plt.show()


def _process_field(field, polygon_data, img_dir, vegetation_index, interval, year):
    try:
        selected_polygon, crop_type, empty_result = _verify_and_filter_field(polygon_data, field)

        if empty_result is not None:
            return empty_result

        # Reproject the polygon to the CRS of the image
        selected_polygon_utm = selected_polygon.to_crs(_extract_projection(img_dir))

        # Extract the EVI data for the current field and the DOY
        mean_evi_values, date_doy = _extract_evi_data(img_dir, selected_polygon_utm)

        # Create a DataFrame with the extracted EVI values
        df = _create_dataframe(mean_evi_values, date_doy, year)

        # Resample the time series to an equal interval and interpolate
        vi_time_serie = (df.loc[:, vegetation_index]
                         .resample(f'{interval}d').max()
                         .interpolate(method="time", limit_direction='both'))

        xdata = np.array(range(vi_time_serie.shape[0]))
        ydata = np.array(vi_time_serie)

        # Initial guess and bounds for [Vb, Va, p, Di, q, Dd]
        p0 = [0.2, 0.6, 0.05, 50 / interval, 0.05, 130 / interval]
        bounds = ([0.0, 0.2, -np.inf, 0, 0, 0],
                  [0.5, 0.8, np.inf, xdata.shape[0], 0.4, xdata.shape[0]])

        # Fit the model
        popt, pcov = opt.curve_fit(_asymmetric_dbl_sigmoid_model,
                                   xdata=xdata,
                                   ydata=ydata,
                                   p0=p0,
                                   bounds=bounds,
                                   method='trf',
                                   maxfev=10000)

        if np.any(np.isinf(pcov)):
            raise RuntimeError("Covariance of the parameters could not be estimated")

        Vb, Va, p, Di, q, Dd = popt

        # Apply the parameters
        vi_fitted = _asymmetric_dbl_sigmoid_model(xdata, *popt)

        # Calculate phenological dates
        D1 = Di + np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
        D2 = Di - np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
        D3 = Dd + np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
        D4 = Dd - np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)

        # Convert dates to the desired format (YYYYMMDD)
        phenological_dates = {
            'field': field,
            'D1': vi_time_serie.index[int(round(D1))].strftime('%Y%m%d'),
            'Di': vi_time_serie.index[int(round(Di))].strftime('%Y%m%d'),
            'D2': vi_time_serie.index[int(round(D2))].strftime('%Y%m%d'),
            'D3': vi_time_serie.index[int(round(D3))].strftime('%Y%m%d'),
            'Dd': vi_time_serie.index[int(round(Dd))].strftime('%Y%m%d'),
            'D4': vi_time_serie.index[int(round(D4))].strftime('%Y%m%d')
        }

        # Optionally plot the phenological dates
        _plot_phenological_dates(df, vi_time_serie, vi_fitted, D1, Di, D2, D3, Dd, D4)

        return phenological_dates

    except IndexError as e:
        print(f"IndexError occurred for field {field}: {e}")
        return {
            'field': field,
            'D1': '',
            'Di': '',
            'D2': '',
            'D3': '',
            'Dd': '',
            'D4': ''
        }

def apply_ads_function(base_dir, polygon_data, station_name, interval, vegetation_index='mean_evi', year='2023', n_jobs=-1):
    '''

     :param base_dir: base dir to look for HLS images
     :param polygon_data: original polygon data with fields information
     :param station_name: station name
     :param interval: interval to resample the time series
     :param vegetation_index: column name of the vegetation index in the polygon data
     :param year: year of the HLS images
     :return: predicted phenological dates
     '''

    # Find all HLS images in the directory
    img_dir = glob.glob(os.path.join(base_dir, '**', station_name, year, 'predicted_img', '*.tif'), recursive=True)

    # Extract field IDs
    fields = polygon_data['field_coun'].unique()

    # Parallel processing using joblib
    results = Parallel(n_jobs=n_jobs)(delayed(_process_field)(field, polygon_data, img_dir, vegetation_index, interval, year) for field in fields)

    return results

def organize_field_data(polygon_data, phenological_dates_df, sowing_model, emergence_model):
    '''

    :param polygon_data: original polygon data with fields information
    :param phenological_dates_df: Predicted phenological dates
    :return: organized field data with predicted phenological dates
    '''
    # Filter out fields with missing phenological dates
    phenological_dates_df = phenological_dates_df.dropna(subset=['D1', 'Di', 'D2', 'D3', 'Dd', 'D4'])
    # Convert columns to datetime format
    date_columns = ['D1', 'Di', 'D2', 'D3', 'Dd', 'D4']
    phenological_dates_df[date_columns] = phenological_dates_df[date_columns].apply(pd.to_datetime, format='%Y%m%d')

    # Extract Day of Year (DOY)
    for col in date_columns:
        phenological_dates_df[col] = phenological_dates_df[col].dt.dayofyear

    # Define a function to apply the models and predict dates
    # Apply the function to the DataFrame
    phenological_dates_df[['sowing', 'emergence']] = phenological_dates_df.apply(
        lambda row: _predict_dates(row, sowing_model, emergence_model) if not row.isnull().any() else pd.Series(
            [np.nan, np.nan]),
        axis=1,
        result_type='expand'
    )

    # Merge the polygon data with the phenological dates
    merged_data = polygon_data.merge(phenological_dates_df, left_on='field_coun', right_on='field', how='left')

    return merged_data


