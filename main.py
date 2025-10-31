import sarsen

input_path: str = (
    "/home/achaad/CGI/ARD4ASAR/asar-xarray/tests/resources/"
    "ASA_IMS_1PNESA20040109_194924_000000182023_00157_09730_0000.N1"
)
cop_dem = "/home/achaad/CGI/ARD4ASAR/DEM/Copernicus_DSM_COG_30_N58_00_E023_00_DEM.tif"
srtm_dem = "/home/achaad/CGI/ARD4ASAR/DEM/srtm_41_01.tif"
product = sarsen.EnvisatProduct(input_path)

gtc = sarsen.envisat_terrain_correction(
    product, dem_urlpath=cop_dem, layers_urlpath="/home/achaad/CGI/ARD4ASAR/envisat_sarsen/layers"
)
