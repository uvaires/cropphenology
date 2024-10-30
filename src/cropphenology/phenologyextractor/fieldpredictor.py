import geopandas as gpd
import pandas as pd
import glob
import os
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
from sklearn.linear_model import ElasticNet
import joblib
import warnings
import warnings
from shapely.errors import ShapelyDeprecationWarning
# Suppress all ShapelyDeprecationWarnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)




# Constants
polygon_path = r'C:\crop_phenology\phenology_field_dates\point_to_build_boulding_box\fields_goodwaterbau2\modified_fields_goodwaterbau2.shp'
planting_model_path = r'C:\crop_phenology\predictions\elastic_net_model_planting.pkl'
emergence_model_path = r'C:\crop_phenology\predictions\elastic_net_model_emergence.pkl'
base_dir = r'C:\crop_phenology'
station_name = 'goodwaterbau2'
year = '2023'
vegetation_index = 'mean_evi'
interval = 1


# Functions

def load_models(planting_model_path, emergence_model_path):
    """Load pre-trained models."""
    planting_model = joblib.load(planting_model_path)
    emergence_model = joblib.load(emergence_model_path)
    assert isinstance(planting_model, ElasticNet) and isinstance(emergence_model, ElasticNet), \
        "Loaded models are not instances of ElasticNet"
    return planting_model, emergence_model


def load_polygon_data(polygon_path):
    """Load polygon data and extract unique field IDs."""
    polygon_data = gpd.read_file(polygon_path)
    fields = polygon_data['field_coun'].unique()
    return polygon_data, fields


def asymmetric_dbl_sigmoid_model(t, Vb, Va, p, Di, q, Dd):
    """Double logistic model function."""
    return Vb + 0.5 * Va * (np.tanh(p * (t - Di)) - np.tanh(q * (t - Dd)))


def convert_doy_to_date(year, doy):
    """Convert Day of Year (DOY) to date in 'YYYYMMDD' format."""
    start_date = pd.to_datetime(f'{year}-01-01')
    target_date = start_date + pd.to_timedelta(doy - 1, unit='d')
    return target_date.strftime('%Y%m%d')


def calculate_mean_evi(img_dir, selected_polygon, target_crs):
    """Calculate mean EVI for each image."""
    selected_polygon_utm = selected_polygon.to_crs(target_crs)
    mean_evi_values, date_doy = [], []
    for img in img_dir:
        date = os.path.basename(img).split('_')[0]
        doy = pd.to_datetime(date, format='%Y%m%d').timetuple().tm_yday
        date_doy.append(doy)
        with rasterio.open(img) as src:
            masked_image, _ = mask(src, selected_polygon_utm.geometry, crop=True)
            mean_value = np.nanmean(masked_image)
            mean_evi_values.append(mean_value if not np.isnan(mean_value) else np.nan)
    return mean_evi_values, date_doy


def fit_model(xdata, ydata, interval):
    """Fit the double sigmoid model to the data."""
    p0 = [0.2, 0.6, 0.05, 50 / interval, 0.05, 130 / interval]
    bounds = ([0.0, 0.2, -np.inf, 0, 0, 0], [0.5, 0.8, np.inf, xdata.shape[0], 0.4, xdata.shape[0]])
    popt, _ = opt.curve_fit(asymmetric_dbl_sigmoid_model, xdata=xdata, ydata=ydata, p0=p0, bounds=bounds,

                        method='trf', maxfev=10000)
    return popt


def calculate_phenological_dates(popt, vi_time_serie):
    """Calculate phenological dates based on fitted parameters."""
    Vb, Va, p, Di, q, Dd = popt
    D1 = Di + np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D2 = Di - np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D3 = Dd + np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D4 = Dd - np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    return {
        'D1': vi_time_serie.index[int(round(D1))].strftime('%Y%m%d'),
        'Di': vi_time_serie.index[int(round(Di))].strftime('%Y%m%d'),
        'D2': vi_time_serie.index[int(round(D2))].strftime('%Y%m%d'),
        'D3': vi_time_serie.index[int(round(D3))].strftime('%Y%m%d'),
        'Dd': vi_time_serie.index[int(round(Dd))].strftime('%Y%m%d'),
        'D4': vi_time_serie.index[int(round(D4))].strftime('%Y%m%d')
    }


def plot_and_save_vi_time_series(df, vi_time_serie, vi_fitted, dates, file_path):
    """Plot and save the VI time series with fitted model and phenological dates."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[vegetation_index], marker='.', lw=0, label='Raw VI')
    ax.plot(vi_time_serie.index, vi_fitted, label=f'Fitted VI')
    ax.set_ylim(0, 1)
    ax.set_ylabel(vegetation_index)
    colors = ['m', 'g', 'yellow', 'c', 'orange', 'b']
    labels = ['D1', 'Di', 'D2', 'D3', 'Dd', 'D4']
    for i, d in enumerate(dates.values()):
        ax.axvline(pd.to_datetime(d, format='%Y%m%d'), 0, vi_fitted[i], color=colors[i], label=f'{labels[i]}: {d}', ls='--')
    ax.legend()
    plt.savefig(file_path)
    plt.show()


def predict_dates(row, planting_model, emergence_model):
    """Predict planting and emergence dates using pre-trained models."""
    features = np.array([[row['D1'], row['Di'], row['D2'], row['D3'], row['Dd'], row['D4']]])
    planting_date = int(round(planting_model.predict(features)[0]))
    emergence_date = int(round(emergence_model.predict(features)[0]))
    return planting_date, emergence_date


def main():
    planting_model, emergence_model = load_models(planting_model_path, emergence_model_path)
    polygon_data, fields = load_polygon_data(polygon_path)
    img_dir = glob.glob(os.path.join(base_dir, '**', station_name, year, 'smoothed_evi', '*.tif'), recursive=True)
    results = []

    for field in fields:
        try:
            selected_polygon = polygon_data[polygon_data['field_coun'] == field]
            crop_type = selected_polygon['crop'].values[0]
            if selected_polygon.empty or crop_type not in ['Corn', 'Soybeans']:
                continue
            with rasterio.open(img_dir[0]) as src:
                mean_evi_values, date_doy = calculate_mean_evi(img_dir, selected_polygon, src.crs)
            df = pd.DataFrame({'date': pd.to_datetime([convert_doy_to_date(year, doy) for doy in date_doy]),
                               vegetation_index: mean_evi_values}).set_index('date')
            vi_time_serie = df[vegetation_index].resample(f'{interval}d').max().interpolate(method="time", limit_direction='both')
            popt = fit_model(np.arange(vi_time_serie.shape[0]), np.array(vi_time_serie), interval)
            dates = calculate_phenological_dates(popt, vi_time_serie)
            results.append({**{'field': field}, **dates})
            file_path = os.path.join(base_dir, 'phenology_field_dates', station_name, f'{station_name}_{year}_field_{field}.png')
            plot_and_save_vi_time_series(df, vi_time_serie, asymmetric_dbl_sigmoid_model(np.arange(vi_time_serie.shape[0]), *popt), dates, file_path)
        except Exception as e:
            print(f"Error occurred for field {field}: {str(e)}")
            continue

    phenological_dates_df = pd.DataFrame(results)
    phenological_dates_df[['planting', 'emergence']] = phenological_dates_df.apply(
        lambda row: predict_dates(row, planting_model, emergence_model) if not row.isnull().any() else pd.Series([np.nan, np.nan]),
        axis=1, result_type='expand')
    merged_data = polygon_data.merge(phenological_dates_df, left_on='field_coun', right_on='field', how='left')
    merged_data.to_file(os.path.join(base_dir, 'phenology_field_dates', station_name, f'phenological_dates_per_field_{station_name}.shp'))


if __name__ == "__main__":
    main()
