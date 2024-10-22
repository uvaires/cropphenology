import pandas as pd
import matplotlib.pyplot as plt
import os
from cropphenology import utils


# Load the data
def plot_phenology(phenology_path:str, base_dir:str, station_name:str, current_year='2023'):
    '''

    :param phenology_path: location of the phenology data
    :param base_dir: output directory
    :param station_name: PhenoCam station name
    :param year: year been processed

    :return: None
    '''
    # Load the data
    phenology_dates = pd.read_csv(phenology_path)

    # Convert dates to DOY
    phenology_dates['Phenocam DOY'] = phenology_dates[' phenocam'].apply(utils._convert_date_to_doy)
    phenology_dates['HLS DOY'] = phenology_dates['hls'].apply(utils._convert_date_to_doy)

    # Define colors and markers for different crops and years
    colors = {
        'D1': 'green',
        'Di': 'blue',
        'D2': 'black',
        'D3': 'purple',
        'D4': 'red',
        'Dd': 'orange',
    }

    markers = {
        2021: 'o',
        2022: 's',
        2023: 'D'
    }

    # Define dictionary to rename crops in the legend
    crop_rename = {
        'D1': 'Greenup',
        'Di': 'MdGreenup',
        'D2': 'Maturity',
        'D3': 'Senescence',
        'D4': 'MdGreendown',
        'Dd': 'Dormancy',
    }

    # Create a single subplot with A4 dimensions (8.27in x 11.69in)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set Times New Roman font
    plt.rcParams["font.family"] = "Times New Roman"

    # Plot HLS Predicted vs. Phenocam
    for crop in phenology_dates['dates'].unique():
        for year in phenology_dates['ano'].unique():
            subset = phenology_dates[(phenology_dates['dates'] == crop) & (phenology_dates['ano'] == year)]
            ax.scatter(subset['Phenocam DOY'], subset['HLS DOY'],
                       color=colors[crop],
                       marker=markers[year],
                       edgecolor='black',
                       s=50,
                       alpha=0.7,
                       label=f'{crop_rename[crop]} {year}')

    rmse_pred, mae_pred, r2_pred, bias_pred = utils._calculate_metrics(phenology_dates['Phenocam DOY'],
                                                                 phenology_dates['HLS DOY'])

    min_val_predicted = min(phenology_dates['Phenocam DOY'].min(), phenology_dates['HLS DOY'].min())
    max_val_predicted = max(phenology_dates['Phenocam DOY'].max(), phenology_dates['HLS DOY'].max())

    # Add some padding to the limits
    padding = 10
    min_limit_predicted = min_val_predicted - padding
    max_limit_predicted = max_val_predicted + padding

    # Plot the 1:1 line
    ax.plot([min_limit_predicted, max_limit_predicted], [min_limit_predicted, max_limit_predicted], color='red',
            linestyle='--', linewidth=2, label='1:1 Line')

    # Set plot limits
    ax.set_xlim(min_limit_predicted, max_limit_predicted)
    ax.set_ylim(min_limit_predicted, max_limit_predicted)

    # Set plot labels and title
    ax.set_xlabel('Phenocam (DOY)', fontsize=14)
    ax.set_ylabel('HLS (DOY)', fontsize=14)
    ax.set_title('a) Phenological stages', fontsize=16, loc='left', fontweight='bold')

    # Enable grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Ensure the aspect ratio is equal
    ax.set_aspect('equal', adjustable='box')

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Add legend under the plot with 3 columns
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=12, ncol=3)

    # Annotate with R², MAE, and RMSE
    textstr_pred = f'R² = {r2_pred:.2f}\nMAE = {mae_pred:.2f}\nRMSE = {rmse_pred:.2f}\nBIAS = {bias_pred:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr_pred, transform=ax.transAxes, fontsize=12, bbox=props, verticalalignment='top')

    # Adjusting the layout to prevent overlap of labels
    plt.tight_layout()
    # Create directories for saving plot and metrics
    save_plot = os.path.join(base_dir, 'data_processed', station_name, current_year, 'plots')
    os.makedirs(save_plot, exist_ok=True)
    plotout_dir = os.path.join(save_plot, 'phenology_dates.png')
    plt.savefig(plotout_dir, dpi=300)
    # Show the plot
    plt.show()


# Plot the phenology stages
def plot_phenology_stages(phenology_path:str, base_dir:str, station_name:str, current_year='2023'):

    '''
   :param phenology_path: location of the phenology data
    :param base_dir: output directory
    :param station_name: PhenoCam station name
    :param year: year been processed

    :return: None

    '''
    phenology_dates = pd.read_csv(phenology_path)

    # Create a dictionary to compare phenocam dates and HLS dates
    dict_dates = {
        'Greenup': ('D1 Phenocam', 'D1 HLS'),
        'MidGreenup': ('Di Phenocam', 'Di HLS'),
        'Maturity': ('D2 Phenocam', 'D2 HLS'),
        'Senescence': ('D3 Phenocam', 'D3 HLS'),
        'MidGreendown': (' Dd Phenocam', 'Dd HLS'),
        'Dormancy': (' D4 Phenocam', 'D4 HLS')
    }

    # Mapping from stage abbreviations to full names
    stage_names = {
        'D1': 'Greenup',
        'Di': 'MidGreenup',
        'D2': 'Maturity',
        'D3': 'Senescence',
        'Dd': 'MidGreendown',
        'D4': 'Dormancy'
    }

    # Define colors and markers for different stages, crops, and years
    stage_colors = {
        'D1': 'green',
        'Di': 'blue',
        'D2': 'black',
        'D3': 'purple',
        'Dd': 'orange',
        'D4': 'red',
    }

    crop_markers = {
        'Soybeans': {
            2021: 'o',
            2022: 's',
            2023: 'D'
        },
        'Corn': {
            2021: 'P',
            2022: 'X',
            2023: '*'
        }
    }

    # Determine the number of rows and columns for the subplots
    num_stages = len(dict_dates)
    num_cols = 2
    num_rows = (num_stages + 1) // num_cols

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 15))  # A4 size in landscape inches

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Titles for subplots
    subplot_titles = ['b)', 'c)', 'd)', 'e)', 'f)', 'g)']

    # Convert dates to DOY outside the loop
    for stage, (phenocam_col, hls_col) in dict_dates.items():
        phenology_dates[f'{phenocam_col} DOY'] = phenology_dates[phenocam_col].apply(utils._convert_date_to_doy)
        phenology_dates[f'{hls_col} DOY'] = phenology_dates[hls_col].apply(utils._convert_date_to_doy)

    for i, (stage, (phenocam_col, hls_col)) in enumerate(dict_dates.items()):
        # Calculate RMSE, MAE, and R²
        rmse, mae, r2, bias = utils._calculate_metrics(phenology_dates[f'{phenocam_col} DOY'],
                                                 phenology_dates[f'{hls_col} DOY'])

        # Plotting
        ax = axes[i]
        for crop in phenology_dates['Crop'].unique():
            for year in phenology_dates['Ano'].unique():
                subset = phenology_dates[(phenology_dates['Crop'] == crop) & (phenology_dates['Ano'] == year)]
                ax.scatter(subset[f'{phenocam_col} DOY'], subset[f'{hls_col} DOY'],
                           color=stage_colors[phenocam_col.split()[0]],
                           marker=crop_markers[crop][year],
                           edgecolor='black',
                           s=50,
                           alpha=0.7,
                           label=f'{crop} {year}')

        min_val = min(phenology_dates[f'{phenocam_col} DOY'].min(), phenology_dates[f'{hls_col} DOY'].min())
        max_val = max(phenology_dates[f'{phenocam_col} DOY'].max(), phenology_dates[f'{hls_col} DOY'].max())
        padding = 10
        min_limit = min_val - padding
        max_limit = max_val + padding

        # Plot the 1:1 line
        ax.plot([min_limit, max_limit], [min_limit, max_limit], color='red', linestyle='--', linewidth=2,
                label='1:1 Line')

        # Set plot limits
        ax.set_xlim(min_limit, max_limit)
        ax.set_ylim(min_limit, max_limit)

        # Set plot labels and title
        ax.set_xlabel(f'Phenocam (DOY)', fontsize=14, fontname='Times New Roman')
        ax.set_ylabel(f'HLS (DOY)', fontsize=14, fontname='Times New Roman')
        ax.set_title(f"{subplot_titles[i]} {stage_names[phenocam_col.split()[0]]}", fontsize=14,
                     fontname='Times New Roman', loc='left', fontweight='bold')

        # Enable grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Ensure the aspect ratio is equal
        ax.set_aspect('equal', adjustable='box')

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        handles[0], handles[1], handles[2], handles[3], handles[4], handles[5] = handles[2], handles[0], handles[1], \
        handles[5], handles[3], handles[4]
        labels[0], labels[1], labels[2], labels[3], labels[4], labels[5] = labels[2], labels[0], labels[1], labels[5], \
        labels[3], labels[4]

        by_label = dict(zip(labels, handles))

        # Adjusted legend with multiple columns
        ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=12)

        # Annotate with R², MAE, and RMSE
        textstr = f'R² = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\nBIAS = {bias:.2f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, bbox=props, verticalalignment='top')

    # Remove empty subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j])

    # Adjust layout and show plot
    plt.tight_layout()
    # Create directories for saving plot and metrics
    save_plot = os.path.join(base_dir, 'data_processed', station_name, current_year, 'plots')
    os.makedirs(save_plot, exist_ok=True)
    plotout_dir = os.path.join(save_plot, 'phenology_stages_plot.png')
    plt.savefig(plotout_dir, dpi=300)
    plt.show()


