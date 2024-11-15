# Cropphenology

This repository implements an operational framework for reconstructing daily EVI image time series from Harmonized Landsat Sentinel-2 images using four gap-filling methods to retrieve crop phenological stages, using the asymmetric double sigmoid. The phenological stages were then used to estimate the sowing and emergence dates of corn and soybeans, validated with field data from PhenoCam observations.

The figure below illustrates the results of the polynomial gap-filling technique to reconstruct the Enhanced Vegetation Index (EVI).
  
![image](https://github.com/user-attachments/assets/62a3aef8-4110-4a12-824c-a1233b0f7dfd)

A good correlation was observed between daily EVI derived from HLS data and PhenoCam data, as presented below: Relation of green chromatic coordinate (GCC) retrieved from PhenoCam and original and predicted EVI derived from HLS images.
                   ![compressed2](https://github.com/user-attachments/assets/6f6f7d33-2509-4d80-8ac2-4de8c4d3c750)

Field predictions show field variability of sowing and emergence dates of corn and soybeans: Predictions of sowing and emerge dates using the elastic net regression model for field-based in IOWA. 
                  ![image](https://github.com/user-attachments/assets/4ee09f39-99b9-4448-87d8-36bd11441acd)



## Dependencies management and package installation
The following command can be used to recreate the conda enviroment with all the dependencies needed to run the code in this repository. This command should be run from the root of the repository.
```
conda env create -f environment.yml
```
If you prefer to use another conda enviroment, you need to activate it and install the package in development mode. To do so, from the repository root, run the command below. It will install the package in development mode, so you can make changes to the code and test it without the need to reinstall the package.
```
pip install -e .
```
You can also install the package directly from GitHub using the following command:
```
pip install git+https://github.com/uvaires/cropphenology
```
