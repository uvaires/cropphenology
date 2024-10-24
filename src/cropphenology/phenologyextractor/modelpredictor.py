import pandas as pd
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from cropphenology import utils
import os
import pickle

def model_predictor(phenology_data: str, base_dir: str, station_name: str, current_year='2023', max_iteration=10000, tol=1e-4,
                    alpha_range=(-4, 1, 6), l1_ratio_range=(0.1, 0.9, 9), C_range=(-2, 2, 5), epsilon_range=(0.1, 1.0, 10)):
    """
    Function to predict the phenology dates using different models and hyperparameters.
    :param phenology_data: path to the phenology data
    :param base_dir: directory to save outputs
    :param station_name: name of the station
    :param current_year: current year for labeling outputs
    :param max_iteration: max number of iterations for ElasticNet
    :param tol: tolerance for ElasticNet
    :param alpha_range: range of alpha values for ElasticNet
    :param l1_ratio_range: range of L1 ratio for ElasticNet
    :param C_range: range of C values for SVM
    :param epsilon_range: range of epsilon for SVM
    :return: NONE
    """

    df = pd.read_csv(phenology_data)

    dict_explanatory_variables = {
        'D1': 'D1 HLS',
        'Di': 'Di HLS',
        'D2': 'D2 HLS',
        'D3': 'D3 HLS',
        'Dd': 'Dd HLS',
        'D4': 'D4 HLS'
    }

    dict_response_variables = {
        'Sowing': 'Sowing Phenocam',
        'Emergence': 'Emergence Phenocam'
    }

    # Columns to select
    columns_to_select = ['Sowing Phenocam', 'Emergence Phenocam', 'D1 HLS', 'Di HLS', 'D2 HLS', 'D3 HLS', 'Dd HLS', 'D4 HLS']
    df_select = df[columns_to_select]

    # Convert dates to day of year (DOY)
    for column in df_select.columns:
        df_select[column] = df_select[column].apply(utils._convert_date_to_doy)

    # Prepare plot
    fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))  # A4 paper size
    titles = ['a) Sowing MLR', 'b) Emergence MLR', 'c) Sowing ENR', 'd) Emergence ENR', 'e) Sowing SVM', 'f) Emergence SVM']

    font = {'fontname': 'Times New Roman'}

    # Define models
    models = {
        "MLR": LinearRegression(),
        "ElasticNet": ElasticNet(max_iter=max_iteration, tol=tol),
        "SVM": SVR()
    }

    # Hyperparameters for ElasticNet and SVM
    params = {
        "ElasticNet": {
            "alpha": np.logspace(alpha_range[0], alpha_range[1], alpha_range[2]),
            "l1_ratio": np.linspace(l1_ratio_range[0], l1_ratio_range[1], l1_ratio_range[2])
        },
        "SVM": {
            "C": np.logspace(C_range[0], C_range[1], C_range[2]),
            "epsilon": np.linspace(epsilon_range[0], epsilon_range[1], epsilon_range[2])
        }
    }

    # Directory to save trained models
    save_model_dir = os.path.join(base_dir, 'data_processed', station_name, current_year, 'models')
    os.makedirs(save_model_dir, exist_ok=True)

    # Loop through models and response variables
    for model_name, model in models.items():
        for i, (response_key, response_value) in enumerate(dict_response_variables.items()):
            X = df_select[dict_explanatory_variables.values()]
            y = df[response_value].apply(utils._convert_date_to_doy)

            if model_name in params:
                model = _tune_model(model, params[model_name], X, y)

            loo = LeaveOneOut()
            predictions, actuals = [], []
            for train_index, test_index in loo.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                predictions.append(y_pred[0])
                actuals.append(y_test.values[0])

            rmse, mae, r2, bias = utils._calculate_metrics(actuals, predictions)
            df['predictions'] = predictions
            df['actuals'] = actuals

            # Save the trained model as a .pkl file
            model_filename = f"{model_name}_{response_key}_model.pkl"
            model_filepath = os.path.join(save_model_dir, model_filename)
            with open(model_filepath, 'wb') as model_file:
                pickle.dump(model, model_file)

            # Plotting scatter plot
            colors = {'Soybeans': 'green', 'Corn': 'yellow'}
            markers = {2023: 'o', 2022: 's', 2021: 'D'}

            ax = axes[list(models.keys()).index(model_name), i]
            for crop in df['Crop'].unique():
                for year in df['Ano'].unique():
                    subset = df[(df['Crop'] == crop) & (df['Ano'] == year)]
                    ax.scatter(subset['actuals'], subset['predictions'],
                               color=colors[crop], marker=markers[year],
                               edgecolor='black', s=50, alpha=0.5,
                               label=f'{crop} {year}')

            min_val = min(df['predictions'].min(), df['actuals'].min())
            max_val = max(df['predictions'].max(), df['actuals'].max())
            padding = 10
            min_limit = min_val - padding
            max_limit = max_val + padding

            ax.plot([min_limit, max_limit], [min_limit + 10, max_limit + 10], color='blue', linestyle='--', linewidth=1,
                    label='+10 days')  # Line for +10 days

            ax.plot([min_limit, max_limit], [min_limit - 10, max_limit - 10], color='green', linestyle='--',
                    linewidth=1, label='-10 days')  # Line for -10 days

            ax.plot([min_limit, max_limit], [min_limit, max_limit], color='red', linestyle='--', linewidth=2,
                    label='1:1 Line')
            ax.set_xlim(min_limit, max_limit)
            ax.set_ylim(min_limit, max_limit)
            ax.set_xlabel(f'Observed {response_key} (DOY)', fontsize=12, **font)  # Adjust font size for A4
            ax.set_ylabel(f'Predicted {response_key} (DOY)', fontsize=12, **font)  # Adjust font size for A4
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_aspect('equal', adjustable='box')

            textstr = f'R² = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\nBIAS = {bias:.2f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.95, 0.05, textstr, fontsize=8, bbox=props, transform=ax.transAxes, verticalalignment='bottom',
                    horizontalalignment='right')  # Adjust font size for A4
            ax.text(-0.1, 1.05, titles[list(models.keys()).index(model_name) * 2 + i], transform=ax.transAxes,
                    fontsize=12, weight='bold', **font, ha='left')

    # Create a single legend for the entire figure
    handles, labels = [], []
    for ax in axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # Adjust order of legend entries
    handles[0], handles[1], handles[2], handles[3], handles[4], handles[5] = handles[2], handles[0], handles[1], handles[5], handles[3], handles[4]
    labels[0], labels[1], labels[2], labels[3], labels[4], labels[5] = labels[2], labels[0], labels[1], labels[5], labels[3], labels[4]

    # Create the legend
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=12, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit legend

    # Save plot
    save_plot = os.path.join(base_dir, 'data_processed', station_name, current_year, 'plots')
    os.makedirs(save_plot, exist_ok=True)
    plotout_dir = os.path.join(save_plot, 'dates_predicted.png')
    plt.savefig(plotout_dir, dpi=300)
    plt.show()

def _tune_model(model, param_grid, X, y):
    """
    Function to tune hyperparameters using a grid search approach.
    :param model: model to be tuned
    :param param_grid: dictionary of hyperparameters
    :param X: explanatory variables
    :param y: response variables
    :return: tuned model
    """
    best_rmse = float('inf')
    best_params = None
    best_model = None

    for params in itertools.product(*param_grid.values()):
        model.set_params(**dict(zip(param_grid.keys(), params)))
        loo = LeaveOneOut()
        predictions, actuals = [], []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.append(y_pred[0])
            actuals.append(y_test.values[0])

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = pickle.loads(pickle.dumps(model))  # Create a deep copy of the model

    if best_params:
        model.set_params(**dict(zip(param_grid.keys(), best_params)))

    return model


# import pandas as pd
# from sklearn.model_selection import LeaveOneOut
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# import itertools
# from sklearn.linear_model import LinearRegression, ElasticNet
# from sklearn.svm import SVR
# from cropphenology import utils
# import os
# import pickle
#
# def model_predictor(phenology_data:str, base_dir:str, station_name:str, current_year='2023', max_iteration=10000, tol=1e-4,
#                     alpha_range=(-4, 1, 6), l1_ratio_range=(0.1, 0.9, 9), C_range=(-2, 2, 5), epsilon_range=(0.1, 1.0, 10)):
#     """
#     Function to predict the phenology dates using different models and hyperparameters.
#     :param phenology_data: path to the phenology data
#     :param base_dir: directory to save outputs
#     :param station_name: name of the station
#     :param current_year: current year for labeling outputs
#     :param max_iteration: max number of iterations for ElasticNet
#     :param tol: tolerance for ElasticNet
#     :param alpha_range: range of alpha values for ElasticNet
#     :param l1_ratio_range: range of L1 ratio for ElasticNet
#     :param C_range: range of C values for SVM
#     :param epsilon_range: range of epsilon for SVM
#     :return: NONE
#     """
#
#     df = pd.read_csv(phenology_data)
#
#     dict_explanatory_variables = {
#         'D1': 'D1 HLS',
#         'Di': 'Di HLS',
#         'D2': 'D2 HLS',
#         'D3': 'D3 HLS',
#         'Dd': 'Dd HLS',
#         'D4': 'D4 HLS'
#     }
#
#     dict_response_variables = {
#         'Sowing': 'Sowing Phenocam',
#         'Emergence': 'Emergence Phenocam'
#     }
#
#     # Columns to select
#     columns_to_select = ['Sowing Phenocam', 'Emergence Phenocam', 'D1 HLS', 'Di HLS', 'D2 HLS', 'D3 HLS', 'Dd HLS',
#                          'D4 HLS']
#     df_select = df[columns_to_select]
#
#     # Convert dates to day of year (DOY)
#     for column in df_select.columns:
#         df_select[column] = df_select[column].apply(utils._convert_date_to_doy)
#
#     # Prepare plot
#     fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))  # A4 paper size
#     titles = ['a) Sowing MLR', 'b) Emergence MLR', 'c) Sowing ENR', 'd) Emergence ENR', 'e) Sowing SVM', 'f) Emergence SVM']
#
#     font = {'fontname': 'Times New Roman'}
#
#     # Define models
#     models = {
#         "MLR": LinearRegression(),
#         "ElasticNet": ElasticNet(max_iter=max_iteration, tol=tol),
#         "SVM": SVR()
#     }
#
#     # Hyperparameters for ElasticNet and SVM
#     params = {
#         "ElasticNet": {
#             "alpha": np.logspace(alpha_range[0], alpha_range[1], alpha_range[2]),
#             "l1_ratio": np.linspace(l1_ratio_range[0], l1_ratio_range[1], l1_ratio_range[2])
#         },
#         "SVM": {
#             "C": np.logspace(C_range[0], C_range[1], C_range[2]),
#             "epsilon": np.linspace(epsilon_range[0], epsilon_range[1], epsilon_range[2])
#         }
#     }
#
#     # Loop through models and response variables
#     for model_name, model in models.items():
#         for i, (response_key, response_value) in enumerate(dict_response_variables.items()):
#             X = df_select[dict_explanatory_variables.values()]
#             y = df[response_value].apply(utils._convert_date_to_doy)
#
#             if model_name in params:
#                 model = _tune_model(model, params[model_name], X, y)
#
#             loo = LeaveOneOut()
#             predictions, actuals = [], []
#             for train_index, test_index in loo.split(X):
#                 X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#                 y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)
#                 predictions.append(y_pred[0])
#                 actuals.append(y_test.values[0])
#
#             rmse, mae, r2, bias = utils._calculate_metrics(actuals, predictions)
#             df['predictions'] = predictions
#             df['actuals'] = actuals
#
#             colors = {'Soybeans': 'green', 'Corn': 'yellow'}
#             markers = {2023: 'o', 2022: 's', 2021: 'D'}
#
#             ax = axes[list(models.keys()).index(model_name), i]
#             for crop in df['Crop'].unique():
#                 for year in df['Ano'].unique():
#                     subset = df[(df['Crop'] == crop) & (df['Ano'] == year)]
#                     ax.scatter(subset['actuals'], subset['predictions'],
#                                color=colors[crop], marker=markers[year],
#                                edgecolor='black', s=50, alpha=0.5,
#                                label=f'{crop} {year}')
#
#             min_val = min(df['predictions'].min(), df['actuals'].min())
#             max_val = max(df['predictions'].max(), df['actuals'].max())
#             padding = 10
#             min_limit = min_val - padding
#             max_limit = max_val + padding
#
#             ax.plot([min_limit, max_limit], [min_limit + 10, max_limit + 10], color='blue', linestyle='--', linewidth=1,
#                     label='+10 days')  # Line for +10 days
#
#             ax.plot([min_limit, max_limit], [min_limit - 10, max_limit - 10], color='green', linestyle='--',
#                     linewidth=1,
#                     label='-10 days')  # Line for -10 days
#
#             ax.plot([min_limit, max_limit], [min_limit, max_limit], color='red', linestyle='--', linewidth=2,
#                     label='1:1 Line')
#             ax.set_xlim(min_limit, max_limit)
#             ax.set_ylim(min_limit, max_limit)
#             ax.set_xlabel(f'Observed {response_key} (DOY)', fontsize=12, **font)  # Adjust font size for A4
#             ax.set_ylabel(f'Predicted {response_key} (DOY)', fontsize=12, **font)  # Adjust font size for A4
#             ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#             ax.set_aspect('equal', adjustable='box')
#
#             textstr = f'R² = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\nBIAS = {bias:.2f}'
#             props = dict(boxstyle='round', facecolor='white', alpha=0.8)
#             ax.text(0.95, 0.05, textstr, fontsize=8, bbox=props, transform=ax.transAxes, verticalalignment='bottom',
#                     horizontalalignment='right')  # Adjust font size for A4
#             ax.text(-0.1, 1.05, titles[list(models.keys()).index(model_name) * 2 + i], transform=ax.transAxes,
#                     fontsize=12,
#                     # Adjust font size for A4
#                     weight='bold', **font, ha='left')
#
#     # Create a single legend for the entire figure
#     handles, labels = [], []
#     for ax in axes.flatten():
#         for handle, label in zip(*ax.get_legend_handles_labels()):
#             if label not in labels:
#                 handles.append(handle)
#                 labels.append(label)
#
#     # Adjust order of legend entries
#     handles[0], handles[1], handles[2], handles[3], handles[4], handles[5] = handles[2], handles[0], handles[1], handles[5], handles[3], handles[4]
#     labels[0], labels[1], labels[2], labels[3], labels[4], labels[5] = labels[2], labels[0], labels[1], labels[5], labels[3], labels[4]
#
#     # Create the legend
#     fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=12, frameon=False)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit legend
#
#     # Save plot
#     save_plot = os.path.join(base_dir, 'data_processed', station_name, current_year, 'plots')
#     os.makedirs(save_plot, exist_ok=True)
#     plotout_dir = os.path.join(save_plot, 'dates_predicted.png')
#     plt.savefig(plotout_dir, dpi=300)
#     plt.show()
#
#
# # Hyperparameter tuning function
# def _tune_model(model, param_grid, X, y):
#     loo = LeaveOneOut()
#     best_params = None
#     best_score = float('inf')
#     for param_combination in [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]:
#         model.set_params(**param_combination)
#         predictions, actuals = [], []
#         for train_index, test_index in loo.split(X):
#             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#             y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             predictions.append(y_pred[0])
#             actuals.append(y_test.values[0])
#         mse = mean_squared_error(actuals, predictions)
#         if mse < best_score:
#             best_score = mse
#             best_params = param_combination
#     model.set_params(**best_params)
#     return model