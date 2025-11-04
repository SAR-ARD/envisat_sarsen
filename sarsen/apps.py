import logging
from typing import Any, Container, Dict, Optional, Tuple
from unittest import mock

import numpy as np
import rioxarray
import xarray as xr
from dask_cuda import LocalCUDACluster, CUDAWorker
from distributed import LocalCluster, SpecCluster, Nanny, Scheduler
import dask.array as da

try:
    import cupy as xp
    import cupy_xarray

    CUDA_MODE = True
except ImportError:
    import numpy as xp

from . import chunking, datamodel, geocoding, orbit, radiometry, scene

logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = 299_792_458.0  # m / s


def make_simulate_acquisition_template(
        template_raster: xr.DataArray,
        correct_radiometry: str | None = None,
        calc_annotation=False,
) -> xr.Dataset:
    acquisition_template = xr.Dataset(
        data_vars={
            "slant_range_time": template_raster,
            "azimuth_time": template_raster.astype("datetime64[ns]"),
        }
    )
    include_variables = {"slant_range_time", "azimuth_time"}
    if calc_annotation:
        include_variables.add("ellipsoid_incidence_angle")
        include_variables.add("local_incidence_angle")
        include_variables.add("gamma_sigma_ratio")
        include_variables.add("layover_shadow_mask")
        acquisition_template["ellipsoid_incidence_angle"] = template_raster
        acquisition_template["local_incidence_angle"] = template_raster
        acquisition_template["gamma_sigma_ratio"] = template_raster
        acquisition_template["layover_shadow_mask"] = template_raster

    if correct_radiometry is not None:
        acquisition_template["gamma_area"] = template_raster
        include_variables.add("gamma_area")

    return acquisition_template


def simulate_acquisition(
        dem_ecef: xr.DataArray,
        orbit_interpolator: orbit.OrbitPolyfitInterpolator,
        include_variables: Container[str] = (),
        azimuth_time: xr.DataArray | float = 0.0,
        **kwargs: Any,
) -> xr.Dataset:
    """Compute the image coordinates of the DEM given the satellite orbit."""
    calc_annotation = "ellipsoid_incidence_angle" in include_variables
    acquisition = geocoding.backward_geocode(
        dem_ecef, orbit_interpolator, calc_annotation=calc_annotation
    )

    # slant_range = (acquisition.dem_distance ** 2).sum(dim="axis") ** 0.5

    slant_range = xr.apply_ufunc(
        lambda a: xp.sqrt(xp.sum(a ** 2, axis=-1)),
        acquisition.dem_distance,
        input_core_dims=[["axis"]],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[acquisition.dem_distance.dtype],
        keep_attrs=True,
    )
    slant_range_time = 2.0 / SPEED_OF_LIGHT * slant_range

    acquisition["slant_range_time"] = slant_range_time

    if include_variables and "gamma_area" in include_variables:
        gamma_area = radiometry.compute_gamma_area(
            dem_ecef, acquisition.dem_distance / slant_range
        )
        acquisition["gamma_area"] = gamma_area

    for data_var_name in acquisition.data_vars:
        if include_variables and data_var_name not in include_variables:
            acquisition = acquisition.drop_vars(data_var_name)  # type: ignore

    # drop coordinates that are not associated with any data variable
    for coord_name in acquisition.coords:
        if all(coord_name not in dv.coords for dv in acquisition.data_vars.values()):
            acquisition = acquisition.drop_vars(coord_name)  # type: ignore

    return acquisition


def map_simulate_acquisition(
        dem_ecef: xr.DataArray,
        orbit_interpolator: orbit.OrbitPolyfitInterpolator,
        template_raster: xr.DataArray | None = None,
        correct_radiometry: str | None = None,
        calc_annotation=False,
        **kwargs: Any,
) -> xr.Dataset:
    if template_raster is None:
        template_raster = dem_ecef.isel(axis=0).drop_vars(["axis", "spatial_ref"]) * 0.0
    acquisition_template = make_simulate_acquisition_template(
        template_raster, correct_radiometry, calc_annotation
    )

    acquisition = xr.map_blocks(
        simulate_acquisition,
        dem_ecef,
        kwargs={
                   "orbit_interpolator": orbit_interpolator,
                   "include_variables": list(acquisition_template.data_vars),
               }
               | kwargs,
        template=acquisition_template,
    )
    return acquisition


def do_terrain_correction(
        product: datamodel.SarProduct,
        dem_raster: xr.DataArray,
        convert_to_dem_ecef_kwargs: dict[str, Any] = {},
        correct_radiometry: str | None = None,
        interp_method: xr.core.types.InterpOptions = "nearest",
        grouping_area_factor: tuple[float, float] = (3.0, 3.0),
        radiometry_chunks: int = 2048,
        radiometry_bound: int = 128,
        seed_step: tuple[int, int] | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    logger.info("pre-process DEM")

    dem_ecef = xr.map_blocks(
        scene.convert_to_dem_ecef, dem_raster, kwargs=convert_to_dem_ecef_kwargs
    )

    logger.info("simulate acquisition")

    template_raster = dem_ecef.isel(axis=0).drop_vars(["axis", "spatial_ref"]) * 0.0

    orbit_interpolator = orbit.OrbitPolyfitInterpolator.from_position(
        product.state_vectors()
    )

    acquisition = map_simulate_acquisition(
        dem_ecef,
        orbit_interpolator,
        correct_radiometry=correct_radiometry,
        seed_step=seed_step,
    )

    simulated_beta_nought = None
    if correct_radiometry is not None:
        logger.info("simulate radiometry")

        grid_parameters = product.grid_parameters(grouping_area_factor)

        if correct_radiometry == "gamma_bilinear":
            gamma_weights = radiometry.gamma_weights_bilinear
        elif correct_radiometry == "gamma_nearest":
            gamma_weights = radiometry.gamma_weights_nearest

        acquisition = acquisition.persist()

        simulated_beta_nought = chunking.map_ovelap(
            obj=acquisition,
            function=gamma_weights,
            chunks=radiometry_chunks,
            bound=radiometry_bound,
            kwargs=grid_parameters,
            template=template_raster,
        )
        simulated_beta_nought.attrs["long_name"] = "terrain-simulated beta nought"

        simulated_beta_nought.x.attrs.update(dem_ecef.x.attrs)
        simulated_beta_nought.y.attrs.update(dem_ecef.y.attrs)
        simulated_beta_nought.rio.write_crs(dem_ecef.rio.crs, inplace=True)

    logger.info("calibrate image")

    beta_nought = product.beta_nought()

    logger.info("terrain-correct image")

    # HACK: we monkey-patch away an optimisation in xr.DataArray.interp that actually makes
    #   the interpolation much slower when indeces are dask arrays.
    with mock.patch("xarray.core.missing._localize", lambda o, i: (o, i)):
        geocoded = product.interp_sar(
            beta_nought,
            azimuth_time=acquisition.azimuth_time,
            slant_range_time=acquisition.slant_range_time,
            method=interp_method,
        )

    if correct_radiometry is not None:
        assert simulated_beta_nought is not None
        geocoded = geocoded / simulated_beta_nought
        geocoded.attrs["long_name"] = "terrain-corrected gamma nought"

    geocoded.x.attrs.update(dem_ecef.x.attrs)
    geocoded.y.attrs.update(dem_ecef.y.attrs)
    geocoded.rio.write_crs(dem_ecef.rio.crs, inplace=True)

    return geocoded, simulated_beta_nought, acquisition


def terrain_correction(
        product: datamodel.SarProduct,
        dem_urlpath: str,
        output_urlpath: str | None = "GTC.tif",
        simulated_urlpath: str | None = None,
        correct_radiometry: str | None = None,
        interp_method: xr.core.types.InterpOptions = "nearest",
        grouping_area_factor: tuple[float, float] = (3.0, 3.0),
        open_dem_raster_kwargs: dict[str, Any] = {},
        chunks: int | None = 1024,
        radiometry_chunks: int = 2048,
        radiometry_bound: int = 128,
        enable_dask_distributed: bool = False,
        client_kwargs: dict[str, Any] = {"processes": False},
        seed_step: tuple[int, int] | None = None,
) -> xr.DataArray:
    """Apply the terrain-correction to sentinel-1 SLC and GRD products.

    :param product: SarProduct instance representing the input data
    :param dem_urlpath: dem path or url
    :param output_urlpath: output path or url
    :param correct_radiometry: default `None`. If `correct_radiometry=None`the radiometric terrain
    correction is not applied. `correct_radiometry=gamma_bilinear` applies the gamma flattening classic
    algorithm using bilinear interpolation to compute the weights. `correct_radiometry=gamma_nearest`
    applies the gamma flattening using nearest neighbours instead of bilinear interpolation.
    'gamma_nearest' significantly reduces the processing time
    :param interp_method: interpolation method for product resampling.
    The interpolation methods are the methods supported by ``xarray.DataArray.interp``
    :param grouping_area_factor: is a tuple of floats greater than 1. The default is `(1, 1)`.
    The `grouping_area_factor`  can be increased (i) to speed up the processing or
    (ii) when the input DEM resolution is low.
    The Gamma Flattening usually works properly if the pixel size of the input DEM is much smaller
    than the pixel size of the input Sentinel-1 product.
    Otherwise, the output may have radiometric distortions.
    This problem can be avoided by increasing the `grouping_area_factor`.
    Be aware that `grouping_area_factor` too high may degrade the final result
    :param open_dem_raster_kwargs: additional keyword arguments passed on to ``xarray.open_dataset``
    to open the `dem_urlpath`
    """
    # rioxarray must be imported explicitly or accesses to `.rio` may fail in dask
    assert rioxarray.__version__

    allowed_correct_radiometry = [None, "gamma_bilinear", "gamma_nearest"]
    if correct_radiometry not in allowed_correct_radiometry:
        raise ValueError(
            f"{correct_radiometry=}. Must be one of: {allowed_correct_radiometry}"
        )
    if simulated_urlpath is not None and correct_radiometry is None:
        raise ValueError("Simulation cannot be saved")
    if output_urlpath is None and simulated_urlpath is None:
        raise ValueError("No output selected")

    allowed_product_types = ["GRD", "SLC"]
    if product.product_type not in allowed_product_types:
        raise ValueError(
            f"{product.product_type=}. Must be one of: {allowed_product_types}"
        )

    output_chunks = chunks if chunks is not None else 512

    to_raster_kwargs: dict[str, Any] = {}
    if enable_dask_distributed:
        from dask.distributed import Client, Lock

        client = Client(**client_kwargs)
        to_raster_kwargs["lock"] = Lock("rio", client=client)
        to_raster_kwargs["compute"] = False
        print(f"Dask distributed dashboard at: {client.dashboard_link}")

    logger.info(f"open DEM {dem_urlpath!r}")

    dem_raster = scene.open_dem_raster(
        dem_urlpath, chunks=chunks, **open_dem_raster_kwargs
    )

    geocoded, simulated_beta_nought, extra_layers = do_terrain_correction(
        product=product,
        dem_raster=dem_raster,
        correct_radiometry=correct_radiometry,
        interp_method=interp_method,
        grouping_area_factor=grouping_area_factor,
        radiometry_chunks=radiometry_chunks,
        radiometry_bound=radiometry_bound,
        seed_step=seed_step,
    )

    if simulated_urlpath is not None:
        assert simulated_beta_nought is not None
        if output_urlpath is not None:
            simulated_beta_nought.persist()

        logger.info("save simulated")

        maybe_delayed = simulated_beta_nought.rio.to_raster(
            simulated_urlpath,
            dtype=np.float32,
            tiled=True,
            blockxsize=output_chunks,
            blockysize=output_chunks,
            compress="ZSTD",
            num_threads="ALL_CPUS",
            **to_raster_kwargs,
        )

        if enable_dask_distributed:
            maybe_delayed.compute()

    if output_urlpath is None:
        assert simulated_beta_nought is not None
        return simulated_beta_nought

    logger.info("save output")

    maybe_delayed = geocoded.rio.to_raster(
        output_urlpath,
        dtype=np.float32,
        tiled=True,
        blockxsize=output_chunks,
        blockysize=output_chunks,
        compress="ZSTD",
        num_threads="ALL_CPUS",
        **to_raster_kwargs,
    )

    if enable_dask_distributed:
        maybe_delayed.compute()

    return geocoded


def envisat_terrain_correction(
        product: datamodel.SarProduct,
        dem_urlpath: str,
        output_urlpath: Optional[str] = "GTC.tif",
        layers_urlpath: Optional[str] = None,
        simulated_urlpath: Optional[str] = None,
        correct_radiometry: Optional[str] = None,
        interp_method: xr.core.types.InterpOptions = "nearest",
        grouping_area_factor: Tuple[float, float] = (3.0, 3.0),
        open_dem_raster_kwargs: Dict[str, Any] = {},
        chunks: Optional[int] = 1024,
        radiometry_chunks: int = 2048,
        radiometry_bound: int = 128,
        enable_dask_distributed: bool = False,
        client_kwargs: Dict[str, Any] = {},
) -> xr.DataArray:
    """Apply the terrain-correction to ENVISAT SLC and GRD products.

    :param product: SarProduct instance representing the input data
    :param dem_urlpath: dem path or url
    :param orbit_group: overrides the orbit group name
    :param calibration_group: overrides the calibration group name
    :param coordinate_conversion_group: overrides the coordinate_conversion group name
    :param output_urlpath: output path or url
    :param correct_radiometry: default `None`. If `correct_radiometry=None`the radiometric terrain
    correction is not applied. `correct_radiometry=gamma_bilinear` applies the gamma flattening classic
    algorithm using bilinear interpolation to compute the weights. `correct_radiometry=gamma_nearest`
    applies the gamma flattening using nearest neighbours instead of bilinear interpolation.
    'gamma_nearest' significantly reduces the processing time
    :param interp_method: interpolation method for product resampling.
    The interpolation methods are the methods supported by ``xarray.DataArray.interp``
    :param multilook: multilook factor. If `None` the multilook is not applied
    :param grouping_area_factor: is a tuple of floats greater than 1. The default is `(1, 1)`.
    The `grouping_area_factor`  can be increased (i) to speed up the processing or
    (ii) when the input DEM resolution is low.
    The Gamma Flattening usually works properly if the pixel size of the input DEM is much smaller
    than the pixel size of the input Sentinel-1 product.
    Otherwise, the output may have radiometric distortions.
    This problem can be avoided by increasing the `grouping_area_factor`.
    Be aware that `grouping_area_factor` too high may degrade the final result
    :param open_dem_raster_kwargs: additional keyword arguments passed on to ``xarray.open_dataset``
    to open the `dem_urlpath`
    """
    # rioxarray must be imported explicitly or accesses to `.rio` may fail in dask
    assert rioxarray.__version__

    allowed_correct_radiometry = [None, "gamma_bilinear", "gamma_nearest"]
    if correct_radiometry not in allowed_correct_radiometry:
        raise ValueError(
            f"{correct_radiometry=}. Must be one of: {allowed_correct_radiometry}"
        )
    if output_urlpath is None and simulated_urlpath is None:
        raise ValueError("No output selected")

    output_chunks = chunks if chunks is not None else 512

    to_raster_kwargs: Dict[str, Any] = {}
    if enable_dask_distributed:
        from dask.distributed import Client, Lock

        # cpu_cluster = LocalCluster(
        #     n_workers=3,
        #     threads_per_worker=2,
        #     processes=True,
        #     dashboard_address=":8787",
        # )
        #
        # # One GPU worker attaches to the same scheduler
        # gpu_cluster = LocalCUDACluster(  # NOTE: this one DOES take scheduler_address
        #     n_workers=1,
        #     scheduler_address=cpu_cluster.scheduler_address,
        #     processes=True,
        #     # env={"CUDA_VISIBLE_DEVICES": "0"},            # optional pinning
        # )

        # combine the two clusters
        gpu_cluster = LocalCUDACluster(
            device_memory_limit="3GB",
            rmm_pool_size="2GB",
        )
        client = Client(gpu_cluster)
        to_raster_kwargs["lock"] = Lock("rio", client=client)  # type: ignore
        to_raster_kwargs["compute"] = False
        print(f"Dask distributed dashboard at: {client.dashboard_link}")

    logger.info(f"open DEM {dem_urlpath!r}")

    dem_raster = scene.open_dem_raster(
        dem_urlpath, chunks=chunks, **open_dem_raster_kwargs
    )

    allowed_product_types = ["GRD", "SLC"]
    if product.product_type not in allowed_product_types:
        raise ValueError(
            f"{product.product_type=}. Must be one of: {allowed_product_types}"
        )

    logger.info("pre-process DEM")

    dem_ecef = xr.map_blocks(
        scene.convert_to_dem_ecef, dem_raster, kwargs={"source_crs": dem_raster.rio.crs}
    )
    dem_ecef = dem_ecef.drop_vars(dem_ecef.rio.grid_mapping).as_cupy()

    logger.info("simulate acquisition")

    template_raster = dem_raster.drop_vars(dem_raster.rio.grid_mapping) * 0.0
    acquisition_template = xr.Dataset(
        data_vars={
            "slant_range_time": template_raster,
            "azimuth_time": template_raster.astype("datetime64[ns]"),
        }
    )

    if product.product_type == "GRD":
        acquisition_template["ground_range"] = template_raster
    if correct_radiometry is not None:
        acquisition_template["gamma_area"] = template_raster

    orbit_interpolator = orbit.OrbitPolyfitInterpolator.from_position(
        product.state_vectors()
    )
    orbit_interpolator.as_cupy()
    template_raster = dem_ecef.isel(axis=0).drop_vars(["axis"]) * 0.0
    acquisition = map_simulate_acquisition(
        dem_ecef,
        orbit_interpolator,
        template_raster,
        correct_radiometry,
        layers_urlpath is not None,
    )

    if correct_radiometry is not None:
        logger.info("simulate radiometry")

        grid_parameters = product.grid_parameters(grouping_area_factor)

        if correct_radiometry == "gamma_bilinear":
            gamma_weights = radiometry.gamma_weights_bilinear
        elif correct_radiometry == "gamma_nearest":
            gamma_weights = radiometry.gamma_weights_nearest

        # acquisition = acquisition.persist()

        simulated_beta_nought = chunking.map_ovelap(
            obj=acquisition,
            function=gamma_weights,
            chunks=radiometry_chunks,
            bound=radiometry_bound,
            kwargs=grid_parameters,
            template=template_raster,
        )
        simulated_beta_nought.x.attrs.update(dem_raster.x.attrs)
        simulated_beta_nought.y.attrs.update(dem_raster.y.attrs)
        simulated_beta_nought.rio.set_crs(dem_raster.rio.crs)

    if simulated_urlpath is not None:
        if output_urlpath is not None:
            simulated_beta_nought.persist()

        logger.info("save simulated")

        maybe_delayed = simulated_beta_nought.rio.to_raster(
            simulated_urlpath,
            dtype=np.float32,
            tiled=True,
            blockxsize=output_chunks,
            blockysize=output_chunks,
            compress="ZSTD",
            num_threads="ALL_CPUS",
            **to_raster_kwargs,
        )

        if enable_dask_distributed:
            maybe_delayed.compute()

    if output_urlpath is None:
        return simulated_beta_nought

    logger.info("calibrate image")
    beta_nought = product.beta_nought()  # xarray DataArray with dims ('azimuth_time','slant_range_time')

    tq_gpu = acquisition.azimuth_time.isel(x=0).data.compute()          # CuPy array on device
    rq_gpu = acquisition.slant_range_time.isel(y=0).data.compute()      # CuPy array on device

    tq_cpu_dt = xp.asnumpy(tq_gpu).astype('datetime64[ns]')              # NumPy on host
    rq_cpu_dt = xp.asnumpy(rq_gpu).astype('datetime64[ns]')              # NumPy on host
    tq_ns_cpu = tq_cpu_dt.view('int64')
    rq_ns_cpu = rq_cpu_dt.view('int64')

    # t_ns  = xp.asarray(beta_nought.azimuth_time.values.astype('datetime64[ns]').view('int64'))
    # r_ns  = xp.asarray(beta_nought.slant_range_time.values.astype('datetime64[ns]').view('int64'))
    # tq_ns = xp.asarray(tq_ns_cpu)
    # rq_ns = xp.asarray(rq_ns_cpu)

    t_ns = xp.asarray(
        beta_nought.azimuth_time.values.astype('datetime64[ns]').view('int64')
    )

    # RANGE: timedelta64 -> int64(ns)
    # (beta_nought slant_range_time should be a duration since transmit/receive, not an absolute time)
    r_ns = xp.asarray(
        beta_nought.slant_range_time.values.astype('timedelta64[ns]').view('int64')
    )

    # New-grid 1-D axes extracted from acquisition:
    tq_gpu = acquisition.azimuth_time.isel(x=0).data.compute()  # device, datetime64
    rq_gpu = acquisition.slant_range_time.isel(y=0).data.compute()  # device, float seconds

    # Host copies with correct kinds
    tq_cpu_dt = xp.asnumpy(tq_gpu).astype('datetime64[ns]')  # datetime64
    tq_ns_cpu = tq_cpu_dt.view('int64')

    # slant_range_time in acquisition is float (seconds). Convert to timedelta64[ns] *first*.
    rq_cpu_td = (xp.asnumpy(rq_gpu).astype(np.float64) * 1e9).astype('timedelta64[ns]')
    rq_ns_cpu = rq_cpu_td.view('int64')

    # Push to GPU
    tq_ns = xp.asarray(tq_ns_cpu)
    rq_ns = xp.asarray(rq_ns_cpu)

    beta_gpu = beta_nought["measurements"].map_blocks(lambda a: xp.asarray(a, dtype=xp.float32))

    def _gpu_interp_block(block, y_old_ns, x_old_ns, y_new_ns, x_new_ns):
        return bilinear_rect_grid(block, y_old_ns, x_old_ns, y_new_ns, x_new_ns)

    Tq = tq_ns_cpu.shape[0]
    Rq = rq_ns_cpu.shape[0]

    with mock.patch("xarray.core.missing._localize", lambda o, i: (o, i)):
        geocoded = xr.apply_ufunc(
            _gpu_interp_block,
            xr.DataArray(beta_gpu, dims=("azimuth_time", "slant_range_time")),
            xr.DataArray(t_ns, dims=("azimuth_time",)),
            xr.DataArray(r_ns, dims=("slant_range_time",)),
            xr.DataArray(tq_ns, dims=("azimuth_time_new",)),
            xr.DataArray(rq_ns, dims=("slant_range_time_new",)),
            input_core_dims=[("azimuth_time", "slant_range_time"),
                             ("azimuth_time",), ("slant_range_time",),
                             ("azimuth_time_new",), ("slant_range_time_new",)],
            output_core_dims=[("azimuth_time_new", "slant_range_time_new")],
            dask="parallelized",
            output_dtypes=[np.float32],
            keep_attrs=True,
            vectorize=False,
            output_sizes={"azimuth_time_new": Tq, "slant_range_time_new": Rq},
        )

    geocoded = geocoded.rename({"azimuth_time_new": "y",
                                "slant_range_time_new": "x"}).assign_coords(
        acquisition.coords
    )
    geocoded.attrs.update(beta_nought.attrs)
    geocoded = geocoded.rio.write_crs(dem_raster.rio.crs)
    geocoded = geocoded.chunk({"x": output_chunks, "y": output_chunks})

    # Back to CPU for write
    if 'CUDA_MODE' in globals() and CUDA_MODE:
        geocoded_cpu = geocoded.copy(deep=False)
        geocoded_cpu.data = geocoded.data.map_blocks(
            lambda a: xp.asnumpy(a), dtype=np.float32
        )
    else:
        geocoded_cpu = geocoded.astype(np.float32)

    maybe_delayed = geocoded_cpu.rio.to_raster(
        output_urlpath,
        dtype=np.float32,
        tiled=True,
        blockxsize=output_chunks,
        blockysize=output_chunks,
        compress="ZSTD",
        num_threads="ALL_CPUS",
        **to_raster_kwargs,
    )
    delayed_layers = []
    # write if urlpath is specified
    if layers_urlpath:
        annotation_layers = [
            "ellipsoid_incidence_angle", "local_incidence_angle",
            "gamma_sigma_ratio", "layover_shadow_mask"
        ]
        layers_to_save = acquisition[[var for var in annotation_layers if var in acquisition]]
        for layer_name, layer_data_array in layers_to_save.data_vars.items():
            layer_filename = f"{layers_urlpath}/{layer_name}.tif"

            delayed_layers.append(layer_data_array.astype(np.float32).rio.to_raster(
                layer_filename,
                tiled=True,
                blockxsize=output_chunks,
                blockysize=output_chunks,
                compress="ZSTD",
                num_threads="ALL_CPUS",
                **to_raster_kwargs,
            ))

    if enable_dask_distributed:
        maybe_delayed.compute()
        for delayed in delayed_layers:
            delayed.compute()

    return geocoded


def slant_range_time_to_ground_range(
        azimuth_time: xr.DataArray, slant_range_time: xr.DataArray
) -> xr.DataArray:
    """Convert slant range time to ground range."""
    return datamodel.GroundRangeSarProduct.slant_range_time_to_ground_range(
        azimuth_time, slant_range_time
    )


def bilinear_rect_grid(data_gpu, y_old_ns, x_old_ns, y_new_ns, x_new_ns):
    """
    data_gpu: 2D CuPy array [Ty, Rx], float32
    y_old_ns: 1D CuPy int64, size Ty (monotonic)
    x_old_ns: 1D CuPy int64, size Rx (monotonic)
    y_new_ns: 1D CuPy int64, size Tq
    x_new_ns: 1D CuPy int64, size Rq
    returns: 2D CuPy float32 [Tq, Rq]
    """

    # Convert to float to avoid int division
    # yi1 = xp.clip(xp.searchsorted(y_old_ns, y_new_ns, side='right') - 1, 0, y_old_ns.size - 2)
    # xi1 = xp.clip(xp.searchsorted(x_old_ns, x_new_ns, side='right') - 1, 0, x_old_ns.size - 2)
    # yi2 = yi1 + 1
    # xi2 = xi1 + 1
    #
    # # Gather neighbors (indices are fine)
    # Y1, X1 = xp.meshgrid(yi1, xi1, indexing='ij')  # [Tq,Rq]
    # Y2, X2 = Y1 + 1, X1 + 1
    #
    # Q11 = data_gpu[Y1, X1]
    # Q12 = data_gpu[Y1, X2]
    # Q21 = data_gpu[Y2, X1]
    # Q22 = data_gpu[Y2, X2]
    #
    # # --- 2) Compute weights from DIFFERENCES, then cast to float ---
    # y0 = y_old_ns[yi1];
    # y1 = y_old_ns[yi2]
    # x0 = x_old_ns[xi1];
    # x1 = x_old_ns[xi2]
    #
    # # work in float but with *small* magnitudes (differences), so no precision loss
    # dy = (y1 - y0).astype(xp.float64)
    # dx = (x1 - x0).astype(xp.float64)
    # num_y = (y_new_ns - y0).astype(xp.float64)
    # num_x = (x_new_ns - x0).astype(xp.float64)
    #
    # wy = xp.where(dy != 0.0, num_y / dy, 0.0)
    # wx = xp.where(dx != 0.0, num_x / dx, 0.0)
    #
    # wy2d = wy[:, None]
    # wx2d = wx[None, :]
    #
    # out = ((1 - wy2d) * (1 - wx2d) * Q11 +
    #        (1 - wy2d) * (wx2d) * Q12 +
    #        (wy2d) * (1 - wx2d) * Q21 +
    #        (wy2d) * (wx2d) * Q22)
    # return out.astype(xp.float32)
    if y_old_ns[1] < y_old_ns[0]:
        y_old_ns = y_old_ns[::-1];
        data_gpu = data_gpu[::-1, :]
    if x_old_ns[1] < x_old_ns[0]:
        x_old_ns = x_old_ns[::-1];
        data_gpu = data_gpu[:, ::-1]

        # 2) Out-of-bounds masks in INT space (no precision loss)
    oob_y = (y_new_ns < y_old_ns[0]) | (y_new_ns > y_old_ns[-1])
    oob_x = (x_new_ns < x_old_ns[0]) | (x_new_ns > x_old_ns[-1])

    # 3) Indices in INT space
    yi1 = xp.clip(xp.searchsorted(y_old_ns, y_new_ns, side='right') - 1, 0, y_old_ns.size - 2)
    xi1 = xp.clip(xp.searchsorted(x_old_ns, x_new_ns, side='right') - 1, 0, x_old_ns.size - 2)
    yi2 = yi1 + 1;
    xi2 = xi1 + 1

    # 4) Gather
    Y1, X1 = xp.meshgrid(yi1, xi1, indexing='ij')
    Q11 = data_gpu[Y1, X1]
    Q12 = data_gpu[Y1, X1 + 1]
    Q21 = data_gpu[Y1 + 1, X1]
    Q22 = data_gpu[Y1 + 1, X1 + 1]

    # 5) Weights from *differences* (small magnitudes) as float
    y0 = y_old_ns[yi1];
    y1 = y_old_ns[yi2]
    x0 = x_old_ns[xi1];
    x1 = x_old_ns[xi2]
    dy = (y1 - y0).astype(xp.float64);
    dx = (x1 - x0).astype(xp.float64)
    wy = xp.where(dy != 0, (y_new_ns - y0).astype(xp.float64) / dy, 0.0)
    wx = xp.where(dx != 0, (x_new_ns - x0).astype(xp.float64) / dx, 0.0)

    wy2d = wy[:, None];
    wx2d = wx[None, :]

    out = ((1 - wy2d) * (1 - wx2d) * Q11 +
           (1 - wy2d) * (wx2d) * Q12 +
           (wy2d) * (1 - wx2d) * Q21 +
           (wy2d) * (wx2d) * Q22).astype(xp.float32)

    # 6) Apply OOB mask → NaN
    if out.dtype != xp.float32:
        out = out.astype(xp.float32)
    mask2d = oob_y[:, None] | oob_x[None, :]
    out = xp.where(mask2d, xp.nan, out)
    return out

