from abc import ABC, abstractmethod
import numpy as np
from datetime import timedelta
from joblib import Parallel, delayed
from geogapfiller import utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor



# Abstract Base Class for Fillers
class Filler(ABC):
    @abstractmethod
    def fill_gaps(self, i, j, raster_values, base_dates) -> np.array:
        pass

# Concrete Class for MedianFiller
class MedianFiller(Filler):
    def __init__(self, window_size=15):
        self.window_size = window_size

    def fill_gaps(self, i, j, raster_values: np.array, base_dates: list):
        raster_values_filled = raster_values.copy()

        for index_ii, raster_value in enumerate(raster_values_filled):
            if np.isnan(raster_value):
                current_date = base_dates[index_ii]
                start_date = current_date - timedelta(days=self.window_size)
                end_date = current_date + timedelta(days=self.window_size)
                valid_indices = [idx for idx, date in enumerate(base_dates) if start_date <= date <= end_date]
                valid_values = [raster_values_filled[idx] for idx in valid_indices if not np.isnan(raster_values_filled[idx])]

                if valid_values:
                    median_value = np.nanmedian(valid_values)
                    raster_values_filled[index_ii] = median_value

        return i, j, raster_values_filled


# Concrete Class for PolynomialFiller
class PolynomialFiller(Filler):
    def __init__(self, poly_degree=2, window_size=15):
        self.poly_degree = poly_degree
        self.window_size = window_size


    def fill_gaps(self, i, j, raster_values: np.array, base_dates: list) -> tuple:
        raster_values_filled = raster_values.copy()

        for index_ii, raster_value in enumerate(raster_values_filled):
            if np.isnan(raster_value):
                current_date = base_dates[index_ii]
                start_date = current_date - timedelta(days=self.window_size)
                end_date = current_date + timedelta(days=self.window_size)

                valid_indices = [idx for idx, date in enumerate(base_dates)
                                 if start_date <= date <= end_date and not np.isnan(raster_values_filled[idx])]
                X_valid = np.array([base_dates[idx].toordinal() for idx in valid_indices]).reshape(-1, 1)
                y_valid = raster_values_filled[valid_indices]

                if len(X_valid) > 1:
                    poly = PolynomialFeatures(self.poly_degree)
                    X_poly = poly.fit_transform(X_valid)

                    model = LinearRegression()
                    model.fit(X_poly, y_valid)

                    X_pred = np.array([[current_date.toordinal()]])
                    pred_value = model.predict(poly.transform(X_pred))


                    raster_values_filled[index_ii] = pred_value[0]

        return i, j, raster_values_filled


# Concrete Class for HarmonicFiller
class HarmonicFiller(Filler):
    def __init__(self, window_size=15):
        self.window_size = window_size

    def fill_gaps(self, i, j, raster_values: np.array, base_dates: list):
        raster_values_filled = raster_values.copy()

        for index_ii, raster_value in enumerate(raster_values_filled):
            if np.isnan(raster_value):
                current_date = base_dates[index_ii]
                start_date = current_date - timedelta(days=self.window_size)
                end_date = current_date + timedelta(days=self.window_size)
                valid_indices = [idx for idx, date in enumerate(base_dates) if start_date <= date <= end_date and not np.isnan(raster_values_filled[idx])]

                if valid_indices:
                    X_valid = np.array([base_dates[idx].toordinal() for idx in valid_indices])
                    y_valid = raster_values_filled[valid_indices]

                    coeffs = self._harmonic_model_fit(X_valid, y_valid)

                    X_pred = np.array([[current_date.toordinal()]])
                    pred_value = self._harmonic_predict(coeffs, X_pred)[0]

                    raster_values_filled[index_ii] = pred_value

        return i, j, raster_values_filled

    def _harmonic_model_fit(self, X, y):
        frequency = 2 * np.pi / 365  # yearly frequency (assuming daily data)
        X_harmonic = np.column_stack([np.ones_like(X), np.sin(frequency * X), np.cos(frequency * X)])
        coeffs, _, _, _ = np.linalg.lstsq(X_harmonic, y, rcond=None)
        return coeffs

    def _harmonic_predict(self, coeffs, X_pred):
        frequency = 2 * np.pi / 365  # yearly frequency
        X_harmonic_pred = np.column_stack([np.ones_like(X_pred), np.sin(frequency * X_pred), np.cos(frequency * X_pred)])
        return np.dot(X_harmonic_pred, coeffs)


# Concrete Class for LightGBMFiller
class LightGBMFiller(Filler):
    def __init__(self, window_size=15, n_estimators=50, random_state=0):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.random_state = random_state


    def fill_gaps(self, i, j, raster_values: np.array, base_dates: list) -> tuple:
        values_filled = raster_values.copy()

        for index_ii, raster_value in enumerate(values_filled):
            if np.isnan(raster_value):
                current_date = base_dates[index_ii]
                start_date = current_date - timedelta(days=self.window_size)
                end_date = current_date + timedelta(days=self.window_size)

                valid_indices = [idx for idx, date in enumerate(base_dates) if start_date <= date <= end_date and not np.isnan(values_filled[idx])]

                if len(valid_indices) > 1:
                    t_valid = np.array([base_dates[idx].toordinal() for idx in valid_indices]).reshape(-1, 1)
                    y_valid = values_filled[valid_indices]

                    lgbm_model = LGBMRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
                    lgbm_model.fit(t_valid, y_valid.ravel())

                    values_filled[index_ii] = lgbm_model.predict(np.array([[current_date.toordinal()]]))[0]

        return i, j, values_filled

def get_method(method_name: str) -> Filler:
    """
    Factory function to return the appropriate filler class based on the method name.
    :param method_name: Name of the gap-filling method (e.g., 'median', 'polynomial', 'harmonic', 'lightgbm')
    :return: Instance of the corresponding Filler subclass
    """
    if method_name == 'median':
        return MedianFiller()
    elif method_name == 'polynomial':
        return PolynomialFiller()
    elif method_name == 'harmonic':
        return HarmonicFiller()
    elif method_name == 'lightgbm':
        return LightGBMFiller()
    else:
        raise ValueError(
            f"Filler {Filler} not implemented. Available methods are: [median, polynomial, harmonic, and 'lightgbm']"
        )

def run_method(filler:Filler, img_list: list, base_dates: list, n_jobs: int = -1)->np.array:
    """
    Run the gap-filling method on the input images
    :param filler: Instance of the Filler subclass
    :param img_list: List of input images
    :param base_dates: List of base dates
    :param n_jobs: Number of jobs to run in parallel
    :return: Filled raster
    """
    # check if the img_list is a list

    # Check if img_list is a list and stack if needed
    if isinstance(img_list, list):
        stacked_raster = utils._stack_raster(img_list)
    elif isinstance(img_list, np.ndarray):
        stacked_raster = img_list  # Skip stacking if it's already a numpy array
    else:
        raise TypeError("img_list must be either a list or a numpy array")

    filled_raster = stacked_raster.copy()

    indices = [(i, j, stacked_raster[:, i, j], base_dates)
               for i in range(stacked_raster.shape[1])
               for j in range(stacked_raster.shape[2])]

    results = Parallel(n_jobs=n_jobs)(delayed(filler.fill_gaps)(i, j, raster_values, base_dates)
                                      for (i, j, raster_values, base_dates) in indices)

    for result in results:
        i, j, raster_values_filled = result
        filled_raster[:, i, j] = raster_values_filled

    return filled_raster