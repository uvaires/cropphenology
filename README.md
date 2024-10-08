# geogapfiller
This repository implements four gap-filling methods to reconstruct geospatial data and predict synthetic images. The methods applied are Polynomial, Median, Harmonic, and LightGBM. Gap-filling is crucial for reconstructing data, as clouds, shadows, and other atmospheric conditions often affect the quality of the images. Description of the methods:

- Median approach: The median is often favored over other statistical measures, such as the mean, because it is less affected by outliers that may result from atmospheric disturbances or sensor errors. This approach selects the median value from the cloud-free pixels in the time series, offering a straightforward solution. However, it does not account for the broader trends or seasonal variations in the data, which may limit its effectiveness in capturing long-term patterns.

- Polynomial approach: The polynomial regression gap-filling approach models the relationship between a dependent variable and one or more independent variables. Higher-degree polynomials can represent more complex relationships, allowing for better data reconstruction. However, as the polynomial degree increases, the model becomes harder to interpret and requires more processing time.

- Harmonic approach: The harmonic gap-filling approach leverages a Fourier-like series, using a combination of sine and cosine functions to estimate missing data. This method is widely applied in remote sensing because of its strength in capturing periodic and seasonal variations. Harmonic models tend to excel when data gaps are evenly distributed, as they can smoothly interpolate across time. However, when gaps are uneven or concentrated in specific periods, the model’s accuracy may decline, resulting in less reliable gap-filling.

- LightGBM approach: The LightGBM gap-filling approach utilizes a tree-based learning algorithm to model relationships in the data. Unlike harmonic models, which assume periodicity, LightGBM does not rely on any inherent patterns and instead learns from the provided training data. It is known for its efficiency and produces results comparable to the Gradient Boosting Machine.
  
![image](https://github.com/user-attachments/assets/62a3aef8-4110-4a12-824c-a1233b0f7dfd)





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
pip install git+https://github.com/uvaires/geogapfiller
```
