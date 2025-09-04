# Software User Manual
This document provides a comprehensive guide on how to use the software effectively.

## Installation
Complete installation instructions are outlined in the {doc}`Setup and Development Guide <sim>` section.

## Python API usage

The ENVISAT .N1 files can be read using the {class}`sarsen.envisat.EnvisatProduct` class. Below is an example of how to read and access data from these files.

```pythonpython
file_path = 'path_to_your_file.N1'
product = sarsen.EnvisatProduct(file_path)
```

### Terrain correction

In order to perform terrain correction, you will need a DEM (Digital Elevation Model) file. Package `elevation` can be used to 
conveniently download SRTM data. The terrain correction can be performed using the {func}`sarsen.apps.envisat_terrain_correction` function.

```python
sarsen.envisat_terrain_correction(product, 
                                  dem_path='path_to_your_dem_file.tif', 
                                  output_urlpath='output_file.tif')
```

The output will be a GeoTIFF file containing the terrain-corrected backscatter values.

### Calibration
Additionally, calibration can be performed by setting `correct_radiometry` attribute of the aforementioned method.

```python
sarsen.envisat_terrain_correction(product, 
                                  dem_path='path_to_your_dem_file.tif', 
                                  output_urlpath='output_file.tif', 
                                  correct_radiometry='gamma_nearest')
```

### Additional layers

Sarsen supports the generation of the following additional layers for the ENVISAT products:
- Local incidence angle
- Ellipsoidal incidence angle
- Layover/shadow mask

In order to generate these layers, you can set the `simulated_urlpath` attribute of the `envisat_terrain_correction` method.
The attribute accepts a path to which the simulated layers will be saved as a multi-band GeoTIFF file.

```python
sarsen.envisat_terrain_correction(product, 
                                  dem_path='path_to_your_dem_file.tif', 
                                  output_urlpath='output_file.tif', 
                                  simulated_urlpath='simulated_layers.tif')
```