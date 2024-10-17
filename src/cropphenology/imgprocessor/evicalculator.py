import os
import glob
import numpy as np
import rasterio
from datetime import datetime


def evi(base_dir, station_name) -> None:
    """
    Processes HLS images, calculates EVI, and exports them to specified directories.

    Args:
        base_dir (str): Root directory containing HLS images

    Returns:
        None
    """

    hls_imgs = glob.glob(os.path.join(base_dir, '**',station_name, '**','hls_cloudless', '**', '*B02*.tif'), recursive=True)

    for hls_img in hls_imgs:
        print(hls_img)
        img_date = os.path.basename(hls_img).split('_')[0]
        convert_date = datetime.strptime(img_date, '%Y%m%d')
        year = convert_date.year
        year = str(year)
        basedir_indices = os.path.join(base_dir,'data_processed', station_name, year, 'spectral_index')
        dir_output_date = os.path.join(basedir_indices)
        os.makedirs(dir_output_date, exist_ok=True)

        blue_band_hls = hls_img
        red_band_hls = hls_img.replace('B02', 'B04')
        nir_band_hls = hls_img.replace('B02', 'NIR')

        blue_band, _ = _read_raster(blue_band_hls)
        del blue_band_hls
        red_band, _ = _read_raster(red_band_hls)
        del red_band_hls
        nir_band, src_profile = _read_raster(nir_band_hls)
        del nir_band_hls

        evi = _calculate_evi(nir_band, red_band, blue_band)
        del blue_band, red_band

        evi_output = os.path.join(dir_output_date, os.path.basename(hls_img).replace('B02', 'evi'))

        # Export the calculated EVI to a GeoTIFF file
        _export_index_to_drive(evi, evi_output, src_profile)


# Helper functions
def _read_raster(raster_path: str):
    """
    Read a raster image and return the raster band and profile.

    Args:
        raster_path (str): Path to the raster image.

    Returns:
        raster_band (np.array): Raster band as a numpy array.
        band_profile (dict): Profile information of the raster band.
    """
    with rasterio.open(raster_path) as src:
        raster_band = src.read(1)
        band_profile = src.profile

    return raster_band, band_profile


def _calculate_evi(nir_band: np.array, red_band: np.array, blue_band: np.array) -> np.array:
    """
    Calculate the Enhanced Vegetation Index (EVI) from the NIR, Red, and Blue bands.

    Args:
        nir_band (np.array): Near-Infrared band.
        red_band (np.array): Red band.
        blue_band (np.array): Blue band.

    Returns:
        evi (np.array): Enhanced Vegetation Index.
    """
    evi_numerator = nir_band - red_band
    evi_denominator = nir_band + 6 * red_band - 7.5 * blue_band + 1
    mask = evi_denominator != 0
    evi = np.zeros_like(nir_band, dtype=np.float32)
    evi[mask] = 2.5 * evi_numerator[mask] / evi_denominator[mask]

    return evi

def _export_index_to_drive(index, output_path, src_profile, data_type='float32'):
    """
    Exports the calculated index to a GeoTIFF file.

    Args:
        index (numpy.ndarray): Index values to be exported.
        output_path (str): Output path for the GeoTIFF file.
        src_profile (dict): Source profile of the raster.

    Returns:
        None
    """
    src_profile['dtype'] = data_type
    src_profile['compress'] = 'lzw'
    src_profile['nodata'] = np.nan
    with rasterio.open(output_path, 'w', **src_profile) as dst:
        dst.write(index, 1)


