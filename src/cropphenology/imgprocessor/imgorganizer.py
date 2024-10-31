from typing import Dict
import os
import glob
from datetime import datetime, timedelta
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio.mask
from rasterio.mask import mask
from shapely.geometry import box

from typing import Dict
import os
import glob
from datetime import datetime, timedelta
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio.mask
from rasterio.mask import mask
from shapely.geometry import box


def organize_hls(hls_raw_images: str, base_dir: str, tile_dir: str, start_date: str, end_date: str,
                 bands_per_product: dict, band_names: dict, phenocam_stations: str = None, station_name: str = None,
                 buffer_distance=None, crop=True) -> None:
    """
    Rename HLS bands, dates, product (Landsat (L30) or Sentinel (S30))
     and organize them in a specified directory structure.

    Args:
        hls_raw_images (str): Folder to get images from.
        base_dir (str): Main output directory for organized HLS bands.
        tile_dir (str): Directory containing tile-specific subdirectories.
        start_date (str): Start date of the range in the format 'YYYYMMDD'.
        end_date (str): End date of the range in the format 'YYYYMMDD'.
        bands_per_product (dict): Dictionary containing the bands to be selected for each HLS product.
        band_names (dict): Dictionary containing the new names for the bands.
        phenocam_stations (str): Path to the shapefile containing the phenocam stations.
        buffer_distance (int): Distance from the PhenoCam station to create the bounding box.
        station_name (str): Name of the phenocam station to be used.
        crop (bool): If True, crop the image to the bounding box; if False, process the entire image.

    :return: None

    """

    # Convert start_date and end_date strings to datetime objects
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')

    # Globbing Input Files
    hls_images = glob.glob(os.path.join(hls_raw_images, tile_dir, '**', '*.tif'), recursive=True)
    hls_tile_image = {file_path: os.path.basename(file_path).split('.')[2] for file_path in hls_images}
    target_crs = _get_valid_crs(hls_images)

    # Create a bounding box around the phenocam station
    interest_region_gdf = _create_bbox(phenocam_stations, station_name, target_crs, buffer_distance)

    # Within the loop where each image file is processed
    for file_path, tile in hls_tile_image.items():

        # Extracting Metadata
        metadata = _extract_metadata_from_path(file_path)

        # Convert metadata date string to datetime
        metadata_date = datetime.strptime(metadata['formatted_date'], '%Y%m%d')
        # Extract the year from the metadata date
        year = metadata_date.year
        # convert year to string
        year = str(year)

        # Check if the date is within the specified range
        if start_date <= metadata_date <= end_date:
            # Extract band name from the file name
            band_name_hls = os.path.basename(file_path).split('.')[6]

            if band_name_hls in bands_per_product.get(metadata['hls_product'], {}):
                # Check if the band name is present in the bands_per_product dictionary for the current HLS product
                if band_name_hls in band_names.get(metadata['hls_product'], {}):
                    # If band name has a specific name defined in band_names, use it for renaming
                    band_name = band_names[metadata['hls_product']][band_name_hls]
                else:
                    # Otherwise, use the original band name
                    band_name = band_name_hls

                # Create a directory to store the data
                dir_output = os.path.join(base_dir, 'data_processed', station_name, year, 'pre_process', 'hls_organized')
                hls_renamed = f"{metadata['formatted_date']}_{metadata['hls_product']}_{tile}_{band_name}.tif"
                dir_output_date = os.path.join(dir_output, metadata['date_string'])
                os.makedirs(dir_output_date, exist_ok=True)

                with rasterio.open(file_path) as src:
                    if crop:
                        raster_bounds = box(*src.bounds)
                        # Check if the raster bounds intersect with the interest region
                        if not interest_region_gdf.geometry.intersects(raster_bounds).any():
                            print(f"Skipping {file_path} as it does not overlap with the interest region.")
                            continue
                        # Clip the raster to the extent of the bounding box
                        clipped_image, clipped_transform = mask(src, interest_region_gdf.geometry, crop=True)
                    else:
                        # If not cropping, load the entire image
                        clipped_image = src.read()
                        clipped_transform = src.transform

                    # Get the metadata of the clipped raster
                    clipped_meta = src.meta.copy()
                    # Update the metadata with the new dimensions and transform
                    clipped_meta.update({"height": clipped_image.shape[1],
                                         "width": clipped_image.shape[2],
                                         "transform": clipped_transform,
                                         "crs": target_crs,
                                         "compress": "lzw"})

                hls_output_path = os.path.join(dir_output_date, hls_renamed)
                # Write the clipped raster to a new file
                with rasterio.open(hls_output_path, 'w', **clipped_meta) as dst:
                    dst.write(clipped_image)

# def organize_hls(hls_raw_images: str, base_dir: str, tile_dir: str, start_date: str, end_date: str,
#                  bands_per_product: dict, band_names: dict, phenocam_stations: str, station_name: str,
#                  buffer_distance=5000) -> None:
#     """
#     Rename HLS bands, dates, product (Landsat (L30) or Sentinel (S30))
#      and organize them in a specified directory structure.
#
#     Args:
#         hls_raw_images (str): Folder to get images from.
#         base_dir (str): Main output directory for organized HLS bands.
#         tile_dir (str): Directory containing tile-specific subdirectories.
#         start_date (str): Start date of the range in the format 'YYYYMMDD'.
#         end_date (str): End date of the range in the format 'YYYYMMDD'.
#         bands_per_product (dict): Dictionary containing the bands to be selected for each HLS product.
#         band_names (dict): Dictionary containing the new names for the bands.
#         phenocam_stations (str): Path to the shapefile containing the phenocam stations.
#         buffer_distance (int): Distance from the PhenoCam station to create the bounding box.
#         station_name (str): Name of the phenocam station to be used.
#
#     :return: None
#
#     """
#
#     # Convert start_date and end_date strings to datetime objects
#     start_date = datetime.strptime(start_date, '%Y%m%d')
#     end_date = datetime.strptime(end_date, '%Y%m%d')
#
#     # Globbing Input Files
#     hls_images = glob.glob(os.path.join(hls_raw_images, tile_dir, '**', '*.tif'), recursive=True)
#     hls_tile_image = {file_path: os.path.basename(file_path).split('.')[2] for file_path in hls_images}
#     target_crs = _get_valid_crs(hls_images)
#
#     # Create a bounding box around the phenocam station
#     interest_region_gdf = _create_bbox(phenocam_stations, station_name, target_crs, buffer_distance)
#
#     # Within the loop where each image file is processed
#     for file_path, tile in hls_tile_image.items():
#
#         # Extracting Metadata
#         metadata = _extract_metadata_from_path(file_path)
#
#         # Convert metadata date string to datetime
#         metadata_date = datetime.strptime(metadata['formatted_date'], '%Y%m%d')
#         # Extract the year from the metadata date
#         year = metadata_date.year
#         # convert year to string
#         year = str(year)
#
#         # Check if the date is within the specified range
#         if start_date <= metadata_date <= end_date:
#             # Extract band name from the file name
#             band_name_hls = os.path.basename(file_path).split('.')[6]
#
#             if band_name_hls in bands_per_product.get(metadata['hls_product'], {}):
#                 # Check if the band name is present in the bands_per_product dictionary for the current HLS product
#                 if band_name_hls in band_names.get(metadata['hls_product'], {}):
#                     # If band name has a specific name defined in band_names, use it for renaming
#                     band_name = band_names[metadata['hls_product']][band_name_hls]
#                 else:
#                     # Otherwise, use the original band name
#                     band_name = band_name_hls
#
#                 # Create a directory to store the data
#                 dir_output = os.path.join(base_dir,'data_processed', station_name, year, 'pre_process', 'hls_organized')
#                 # Continue processing only if the date is within the range
#                 hls_renamed = f"{metadata['formatted_date']}_{metadata['hls_product']}_{tile}_{band_name}.tif"
#                 dir_output_date = os.path.join(dir_output, metadata['date_string'])
#                 os.makedirs(dir_output_date, exist_ok=True)
#
#                 with rasterio.open(file_path) as src:
#                     raster_bounds = box(*src.bounds)
#                     # Check if the raster bounds intersect with the interest region
#                     if not interest_region_gdf.geometry.intersects(raster_bounds).any():
#                         # Skip this image if it does not overlap
#                         print(f"Skipping {file_path} as it does not overlap with the interest region.")
#                         continue
#                     # Clip the raster to the extent of the bounding box
#                     clipped_image, clipped_transform = mask(src, interest_region_gdf.geometry, crop=True)
#                     # Get the metadata of the clipped raster
#                     clipped_meta = src.meta.copy()
#
#                 # Update the metadata with the new dimensions and transform
#                 clipped_meta.update({"height": clipped_image.shape[1],
#                                      "width": clipped_image.shape[2],
#                                      "transform": clipped_transform,
#                                      "crs": target_crs,
#                                      "compress": "lzw"})
#
#                 hls_output_path = os.path.join(dir_output_date, hls_renamed)
#                 # Write the clipped raster to a new file
#                 with rasterio.open(hls_output_path, 'w', **clipped_meta) as dst:
#                     dst.write(clipped_image)


### Privite Functions ###
def _is_valid_utm_crs(crs):
    """
    Check if the provided CRS is a valid UTM projection.

    Args:
        crs (rasterio.crs.CRS): Coordinate Reference System.

    Returns:
        bool: True if the CRS is a valid UTM projection, False otherwise.
    """
    try:
        epsg_code = crs.to_epsg()
        return 32610 <= epsg_code <= 32619
    except:
        return False


def _extract_metadata_from_path(file_path: str) -> Dict[str, str]:
    """
    Find and return the valid UTM CRS among a list of HLS image paths.

    Args:
        hls_images_paths (List[str]): List of paths to HLS images.

    Returns:
        str: Valid UTM CRS in the format 'EPSG:xxxx' if found, otherwise an empty string.
    """
    julian_date = os.path.basename(file_path).split('.')[3].split('T')[0]
    year = int(julian_date[:4])
    day_of_year = int(julian_date[4:])
    image_date = datetime(year, 1, 1) + timedelta(day_of_year - 1)
    date_string = image_date.strftime('%Y%m%d')
    formatted_date = image_date.strftime('%Y%m%d')
    hls_product = os.path.basename(file_path).split('.')[1]
    metadata = {
        'date_string': date_string,
        'formatted_date': formatted_date,
        'hls_product': hls_product,
    }
    return metadata


def _get_valid_crs(hls_images_paths) -> str:
    """
        Find and return the valid UTM CRS among a list of HLS image paths.

        Args:
            hls_images_paths (List[str]): List of paths to HLS images.

        Returns:
            str: Valid UTM CRS in the format 'EPSG:xxxx' if found, otherwise an empty string.
        """
    for path in hls_images_paths:
        with rasterio.open(path) as src:
            crs = src.crs
        if _is_valid_utm_crs(crs):
            return crs


def _create_bbox(phenocam_stations, station_name, target_crs, buffer_distance=50):
    ''' Create a bounding box around the phenocam station
    Args:
        phenocam_stations (str): Path to the shapefile containing the phenocam stations.
        station_name (str): Name of the phenocam station to be used.
        target_crs (str): Target CRS to reproject the bounding box.
        buffer_distance (int): Buffer distance around the station in meters

        Returns: bounding box around the phenocam station

        '''

    # Load the shapefile
    gdf = gpd.read_file(phenocam_stations)
    point = gdf[gdf['stations'] == station_name]
    point_reprojected = point.to_crs(target_crs)

    point = point_reprojected.geometry.values[0]
    minx, miny, maxx, maxy = point.x - buffer_distance, point.y - buffer_distance, point.x + buffer_distance, point.y + buffer_distance
    bbox = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
    bbox = gpd.GeoDataFrame(geometry=[bbox], crs=target_crs)

    return bbox



