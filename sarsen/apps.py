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
        solution = 1
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
    beta_nought = product.beta_nought()

    if solution == 1:
        # SOLUTION 1
        acquisition = acquisition.chunk({"y": output_chunks, "x": output_chunks})

        tq_gpu = acquisition.azimuth_time.data.compute()  # shape (Y, X), datetime64 on GPU
        tq_ns = xr.apply_ufunc(
            lambda a: a.astype("datetime64[ns]").astype("int64"),
            acquisition.azimuth_time,
            dask="parallelized", output_dtypes=[np.int64]
        ).chunk({"y": output_chunks, "x": output_chunks})

        def to_td_ns_int(a):
            if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.integer):
                return (a.astype(np.float64) * 1e9).astype(np.int64)
            return a.astype("timedelta64[ns]").astype("int64")

        rq_ns = xr.apply_ufunc(
            to_td_ns_int, acquisition.slant_range_time,
            dask="parallelized", output_dtypes=[np.int64]
        ).chunk({"y": output_chunks, "x": output_chunks})

        beta_da = beta_nought["measurements"].chunk({"azimuth_time": output_chunks, "slant_range_time": output_chunks})

        if 'CUDA_MODE' in globals() and CUDA_MODE:
            beta_gpu = xr.apply_ufunc(
                lambda a: xp.asarray(a, dtype=xp.float32),
                beta_da,
                dask="parallelized",
                output_dtypes=[np.float32],
            )
        else:
            beta_gpu = beta_da.astype(np.float32)

        with mock.patch("xarray.core.missing._localize", lambda o, i: (o, i)):
            geocoded = xr.apply_ufunc(
                lambda block, t_old, r_old, tq_block, rq_block:
                bilinear_rect_grid_2d_block(block, t_old, r_old, tq_block, rq_block),
                beta_gpu,
                xr.DataArray(xp.asarray(beta_nought.azimuth_time.values.astype("datetime64[ns]").view("int64")),
                             dims=("azimuth_time",)),
                xr.DataArray(xp.asarray(beta_nought.slant_range_time.values.astype("timedelta64[ns]").view("int64")),
                             dims=("slant_range_time",)),
                tq_ns, rq_ns,
                input_core_dims=[("azimuth_time", "slant_range_time"),
                                 ("azimuth_time",), ("slant_range_time",),
                                 ("y", "x"), ("y", "x")],
                output_core_dims=[("y", "x")],
                dask="parallelized",
                output_dtypes=[np.float32],
                vectorize=False,
                dask_gufunc_kwargs={"allow_rechunk": True},
            ).assign_coords(y=acquisition.y, x=acquisition.x).rio.write_crs(dem_raster.rio.crs)

        # END OF SOLUTION 1
    if solution == 2:
        # SOLUTION 2:
        beta_nought = product.beta_nought()

        beta_gpu_da = beta_nought["measurements"].map_blocks(
            lambda a: xp.asarray(a, dtype=xp.float32)
        )

        t_ns_raw = xp.asarray(beta_nought.azimuth_time.values.astype('datetime64[ns]').view('int64'), dtype=xp.float64)
        t_ns_da = xr.DataArray(t_ns_raw, dims=["azimuth_time"], coords={"azimuth_time": beta_nought.azimuth_time})
        r_ns_raw = xp.asarray(beta_nought.slant_range_time.values.astype('datetime64[ns]').view('int64'),
                              dtype=xp.float64)
        r_ns_da = xr.DataArray(r_ns_raw, dims=["slant_range_time"],
                               coords={"slant_range_time": beta_nought.slant_range_time})

        tq_ns_map_da = xr.apply_ufunc(
            lambda x: x.view('int64').astype(xp.float64),
            acquisition.azimuth_time,
            dask="allowed", keep_attrs=True
        )

        rq_ns_map_da = xr.apply_ufunc(
            lambda x: x.astype(xp.float64),
            acquisition.slant_range_time,
            dask="allowed", keep_attrs=True
        )

        def _gpu_remap(data_block, t_coord_ns, r_coord_ns, t_lookup_map, r_lookup_map):
            """
            Performs 2D remapping using CuPy.

            data_block: 2D CuPy array (radar data)
            t_coord_ns: 1D CuPy array (azimuth time coordinate in ns)
            r_coord_ns: 1D CuPy array (slant range time coordinate in ns)
            t_lookup_map: 2D CuPy array (azimuth time lookup table)
            r_lookup_map: 2D CuPy array (slant range time lookup table)
            """
            from cupyx.scipy.ndimage import map_coordinates

            t_indices = xp.arange(t_coord_ns.size, dtype=xp.float64)
            r_indices = xp.arange(r_coord_ns.size, dtype=xp.float64)

            original_shape = t_lookup_map.shape

            indices_y_flat = xp.interp(t_lookup_map.flatten(), t_coord_ns, t_indices,
                                       left=xp.nan, right=xp.nan)
            indices_y = indices_y_flat.reshape(original_shape)

            indices_x_flat = xp.interp(r_lookup_map.flatten(), r_coord_ns, r_indices,
                                       left=xp.nan, right=xp.nan)
            indices_x = indices_x_flat.reshape(original_shape)

            indices = xp.stack([indices_y, indices_x], axis=0)

            nan_mask = xp.isnan(indices[0]) | xp.isnan(indices[1])

            indices[xp.isnan(indices)] = 0

            geocoded_block = map_coordinates(data_block, indices,
                                             order=1,
                                             mode='constant',
                                             cval=xp.nan)

            geocoded_block[nan_mask] = xp.nan

            return geocoded_block.astype(xp.float32)

        logger.info("terrain-correct image")

        with mock.patch("xarray.core.missing._localize", lambda o, i: (o, i)):
            geocoded = xr.apply_ufunc(
                _gpu_remap,
                beta_gpu_da,
                t_ns_da,
                r_ns_da,
                tq_ns_map_da,
                rq_ns_map_da,
                input_core_dims=[
                    ['azimuth_time', 'slant_range_time'],
                    ['azimuth_time'],
                    ['slant_range_time'],
                    ['y', 'x'],
                    ['y', 'x']
                ],
                output_core_dims=[['y', 'x']],
                dask="parallelized",
                output_dtypes=[np.float32],
                keep_attrs=True,
                dask_gufunc_kwargs={'allow_rechunk': True}
            )

        # END OF SOLUTION 2
    geocoded = geocoded.assign_coords(acquisition.coords)
    geocoded.attrs.update(beta_nought.attrs)
    geocoded = geocoded.rio.write_crs(dem_raster.rio.crs)
    geocoded = geocoded.chunk({"x": output_chunks, "y": output_chunks})

    if 'CUDA_MODE' in globals() and CUDA_MODE:
        geocoded_cpu = geocoded.copy(deep=False)
        geocoded_cpu.data = geocoded.data.map_blocks(
            lambda a: xp.asnumpy(a), dtype=np.float32
        )
    else:
        geocoded_cpu = geocoded.astype(np.float32)
    coords_to_drop = []
    for coord_name, coord_val in geocoded_cpu.coords.items():
        # Drop coordinates that are not 1D dimensions (like 'x', 'y')
        # and are not 'spatial_ref'
        if coord_name not in geocoded_cpu.dims and coord_name != 'spatial_ref':
            coords_to_drop.append(coord_name)

    if coords_to_drop:
        logger.info(f"Dropping non-spatial coordinates before saving: {coords_to_drop}")
        geocoded_cpu = geocoded_cpu.drop_vars(coords_to_drop)
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


def bilinear_rect_grid_2d(data_gpu, y_old_ns, x_old_ns, tq_ns, rq_ns):
    # Ensure increasing axes
    if y_old_ns[1] < y_old_ns[0]:
        y_old_ns = y_old_ns[::-1];
        data_gpu = data_gpu[::-1, :]
    if x_old_ns[1] < x_old_ns[0]:
        x_old_ns = x_old_ns[::-1];
        data_gpu = data_gpu[:, ::-1]

    # OOB mask in int space
    oob = (tq_ns < y_old_ns[0]) | (tq_ns > y_old_ns[-1]) | \
          (rq_ns < x_old_ns[0]) | (rq_ns > x_old_ns[-1])

    # Indices in int space (searchsorted supports array-like)
    yi1 = xp.clip(xp.searchsorted(y_old_ns, tq_ns, side='right') - 1, 0, y_old_ns.size - 2)
    xi1 = xp.clip(xp.searchsorted(x_old_ns, rq_ns, side='right') - 1, 0, x_old_ns.size - 2)
    yi2 = yi1 + 1;
    xi2 = xi1 + 1

    # Neighbor values (advanced indexing with 2-D index arrays)
    Q11 = data_gpu[yi1, xi1]
    Q12 = data_gpu[yi1, xi2]
    Q21 = data_gpu[yi2, xi1]
    Q22 = data_gpu[yi2, xi2]

    # Weights from *differences* (keep numbers small → stable)
    y0 = y_old_ns[yi1];
    y1 = y_old_ns[yi2]
    x0 = x_old_ns[xi1];
    x1 = x_old_ns[xi2]
    dy = (y1 - y0).astype(xp.float64)
    dx = (x1 - x0).astype(xp.float64)
    wy = xp.where(dy != 0, (tq_ns - y0).astype(xp.float64) / dy, 0.0)
    wx = xp.where(dx != 0, (rq_ns - x0).astype(xp.float64) / dx, 0.0)

    out = ((1 - wy) * (1 - wx) * Q11 +
           (1 - wy) * (wx) * Q12 +
           (wy) * (1 - wx) * Q21 +
           (wy) * (wx) * Q22).astype(xp.float32)

    # Mask OOB → NaN, not edge clamping
    return xp.where(oob, xp.nan, out)

def bilinear_rect_grid_2d_block(data_block, y_old_ns, x_old_ns, tq_blk, rq_blk):
    img = xp.asarray(data_block, dtype=xp.float32)  # [Ty,Rx]
    y_old = xp.asarray(y_old_ns, dtype=xp.int64)  # [Ty]
    x_old = xp.asarray(x_old_ns, dtype=xp.int64)  # [Rx]
    tq = xp.asarray(tq_blk, dtype=xp.int64)  # [H,W]  (was CPU)
    rq = xp.asarray(rq_blk, dtype=xp.int64)  # [H,W]

    # Ensure increasing axes (flip without extra copies)
    if y_old.size > 1 and y_old[1] < y_old[0]:
        img = img[::-1, :];
        y_old = y_old[::-1]
    if x_old.size > 1 and x_old[1] < x_old[0]:
        img = img[:, ::-1];
        x_old = x_old[::-1]

    # OOB mask
    oob = (tq < y_old[0]) | (tq > y_old[-1]) | (rq < x_old[0]) | (rq > x_old[-1])

    # Indices (clip) – cast to int32 right away to save memory
    yi1 = xp.clip(xp.searchsorted(y_old, tq, side="right") - 1, 0, y_old.size - 2).astype(xp.int32)
    xi1 = xp.clip(xp.searchsorted(x_old, rq, side="right") - 1, 0, x_old.size - 2).astype(xp.int32)
    yi2 = yi1 + 1;
    xi2 = xi1 + 1

    # Neighbors horizontally & horizontal blend (first pass)
    # Gather left/right
    left = img[yi1, xi1]  # Q11
    right = img[yi1, xi2]  # Q12

    x0 = x_old[xi1].astype(xp.int64);
    x1 = x_old[xi2].astype(xp.int64)
    dx = (x1 - x0).astype(xp.float32)
    nx = (rq - x0).astype(xp.float32)
    wx = xp.where(dx != 0.0, nx / dx, 0.0).astype(xp.float32)

    # Horizontal blend top row: T = (1-wx)*left + wx*right
    top = (left * (1.0 - wx) + right * wx).astype(xp.float32)
    del left, right, x0, x1, dx, nx, wx  # free intermediates

    # Second row neighbors and horizontal blend
    left2 = img[yi2, xi1]  # Q21
    right2 = img[yi2, xi2]  # Q22

    x0 = x_old[xi1].astype(xp.int64);
    x1 = x_old[xi2].astype(xp.int64)
    dx = (x1 - x0).astype(xp.float32)
    nx = (rq - x0).astype(xp.float32)
    wx = xp.where(dx != 0.0, nx / dx, 0.0).astype(xp.float32)

    bottom = (left2 * (1.0 - wx) + right2 * wx).astype(xp.float32)
    del left2, right2, x0, x1, dx, nx, wx

    # Vertical weights wy in float32
    y0 = y_old[yi1].astype(xp.int64);
    y1 = y_old[yi2].astype(xp.int64)
    dy = (y1 - y0).astype(xp.float32)
    ny = (tq - y0).astype(xp.float32)
    wy = xp.where(dy != 0.0, ny / dy, 0.0).astype(xp.float32)

    # Final vertical blend (second pass)
    out = (top * (1.0 - wy) + bottom * wy).astype(xp.float32)
    del top, bottom, y0, y1, dy, ny, wy

    # Apply OOB mask (NaN outside)
    return xp.where(oob, xp.nan, out)
