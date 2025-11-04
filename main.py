import sarsen
import rasterio
import rasterio.mask
from shapely.geometry import box

input_path: str = (
    "/home/achaad/CGI/ARD4ASAR/asar-xarray/tests/resources/"
    "ASA_IMS_1PNESA20040109_194924_000000182023_00157_09730_0000.N1"
)
cop_dem = "/home/achaad/CGI/ARD4ASAR/DEM/Copernicus_DSM_COG_30_N58_00_E023_00_DEM.tif"
srtm_dem = "/home/achaad/CGI/ARD4ASAR/DEM/srtm_41_01.tif"
product = sarsen.EnvisatProduct(input_path)

def clip_dem():
    bbox = box(23, 58.5, 24, 59.5)
    geometries = [bbox]
    with rasterio.open(srtm_dem) as src:
        print(src.crs)
        print(src.bounds)
        out_image, out_transform = rasterio.mask.mask(src, geometries, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],"transform": out_transform})

        with rasterio.open("cropped_dem.tif", "w", **out_meta) as dest:
            dest.write(out_image)

# clip_dem()

cropped_dem = "/home/achaad/CGI/ARD4ASAR/envisat_sarsen/cropped_dem.tif"

gtc = sarsen.envisat_terrain_correction(
    product, dem_urlpath=srtm_dem,
    output_urlpath="/home/achaad/CGI/ARD4ASAR/envisat_sarsen/output.tif",
    solution=1,
    chunks=512
    # layers_urlpath="/home/achaad/CGI/ARD4ASAR/envisat_sarsen/layers"
)
