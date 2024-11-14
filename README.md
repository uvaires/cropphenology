# Cropphenology

This repository implements an operational framework for reconstructing daily EVI image time series from Harmonized Landsat Sentinel-2 images using four gap-filling methods to retrieve crop phenological stages, using the asymmetric double sigmoid. The phenological stages were then used to estimate the sowing and emergence dates of corn and soybeans, validated with field data from PhenoCam observations.

The figure below illustrates the results of the polynomial gap-filling technique to reconstruct the Enhanced Vegetation Index (EVI).
  
![image](https://github.com/user-attachments/assets/62a3aef8-4110-4a12-824c-a1233b0f7dfd)

A good correlation was observed between HLS data and PhenoCam data, as presented below:




## Dependencies management and package installation
The following command can be used to recreate the conda enviroment with all the dependencies needed to run the code in this repository. The package is also installed in development mode. This command should be run from the root of the repository.
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
