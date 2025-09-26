import sarsen

input_path: str = "C:/Users/peran/CGI/SAR/input/resources_ASA_IMS_1PNESA20040109_194924_000000182023_00157_09730_0000.N1"
cop_dem = "C:/Users/peran/CGI/SAR/dem/Copernicus_DSM_COG_30_N58_00_E023_00_DEM.tif"
srtm_dem = "C:/Users/peran/CGI/SAR/dem/srtm_41_01.tif"
product = sarsen.EnvisatProduct(input_path)

gtc = sarsen.envisat_terrain_correction(product, dem_urlpath=srtm_dem, layers_urlpath="layers.tif")