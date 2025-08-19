"""Reference "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019.

See: https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3
"""
import datetime
import functools
import math
from typing import Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt
import xarray as xr

from . import orbit

ArrayLike = TypeVar("ArrayLike", bound=npt.ArrayLike)
FloatArrayLike = TypeVar("FloatArrayLike", bound=npt.ArrayLike)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def secant_method(
        ufunc: Callable[[ArrayLike], tuple[FloatArrayLike, Any]],
        t_prev: ArrayLike,
        t_curr: ArrayLike,
        diff_ufunc: float = 1.0,
        diff_t: Any = 1e-6,
        maxiter: int = 10,
) -> tuple[ArrayLike, ArrayLike, FloatArrayLike, int, Any]:
    """Return the root of ufunc calculated using the secant method."""
    # implementation modified from https://en.wikipedia.org/wiki/Secant_method
    f_prev, _ = ufunc(t_prev)

    # strong convergence, all points below one of the two thresholds
    for k in range(maxiter):
        f_curr, payload_curr = ufunc(t_curr)

        # print(f"{f_curr / 7500}")

        # the `not np.any` construct let us accept `np.nan` as good values
        if not np.any((np.abs(f_curr) > diff_ufunc)):
            break

        t_diff = t_curr - t_prev  # type: ignore

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

        q = f_curr - f_prev  # type: ignore

        # NOTE: in same cases f_curr * t_diff overflows datetime64[ns] before the division by q
        t_prev, t_curr = t_curr, t_curr - np.where(q != 0, f_curr / q, 0) * t_diff  # type: ignore
        f_prev = f_curr

    return t_curr, t_prev, f_curr, k, payload_curr


def newton_raphson_method(
        ufunc: Callable[[ArrayLike], tuple[FloatArrayLike, Any]],
        ufunc_prime: Callable[[ArrayLike, Any], FloatArrayLike],
        t_curr: ArrayLike,
        diff_ufunc: float = 1.0,
        diff_t: Any = 1e-6,
        maxiter: int = 10,
) -> tuple[ArrayLike, FloatArrayLike, int, Any]:
    """Return the root of ufunc calculated using the Newton method."""
    # implementation based on https://en.wikipedia.org/wiki/Newton%27s_method
    # strong convergence, all points below one of the two thresholds
    for k in range(maxiter):
        f_curr, payload_curr = ufunc(t_curr)

        # print(f"{f_curr / 7500}")

        # the `not np.any` construct let us accept `np.nan` as good values
        if not np.any((np.abs(f_curr) > diff_ufunc)):
            break

        fp_curr = ufunc_prime(t_curr, payload_curr)

        t_diff = f_curr / fp_curr  # type: ignore

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

        t_curr = t_curr - t_diff  # type: ignore

    return t_curr, f_curr, k, payload_curr


def zero_doppler_plane_distance_velocity(
        dem_ecef: xr.DataArray,
        orbit_interpolator: orbit.OrbitPolyfitInterpolator,
        orbit_time: xr.DataArray,
        dim: str = "axis",
) -> tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    sar_ecef = orbit_interpolator.position_from_orbit_time(orbit_time)
    dem_distance = dem_ecef - orbit_interpolator.position_from_orbit_time(orbit_time)
    satellite_velocity = orbit_interpolator.velocity_from_orbit_time(orbit_time)
    plane_distance_velocity = (dem_distance * satellite_velocity).sum(dim, skipna=False)
    return plane_distance_velocity, (dem_distance, satellite_velocity, sar_ecef)


def zero_doppler_plane_distance_velocity_prime(
        orbit_interpolator: orbit.OrbitPolyfitInterpolator,
        orbit_time: xr.DataArray,
        payload: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
        dim: str = "axis",
) -> xr.DataArray:
    dem_distance, satellite_velocity, sar_ecef = payload

    plane_distance_velocity_prime = (
            dem_distance * orbit_interpolator.acceleration_from_orbit_time(orbit_time)
            - satellite_velocity ** 2
    ).sum(dim)
    return plane_distance_velocity_prime


def backward_geocode_simple(
        dem_ecef: xr.DataArray,
        orbit_interpolator: orbit.OrbitPolyfitInterpolator,
        orbit_time_guess: xr.DataArray | float = 0.0,
        dim: str = "axis",
        zero_doppler_distance: float = 1.0,
        satellite_speed: float = 7_500.0,
        method: str = "secant",
        orbit_time_prev_shift: float = -0.1,
        maxiter: int = 10,
        calc_annotation=False,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    diff_ufunc = zero_doppler_distance * satellite_speed
    zero_doppler = functools.partial(
        zero_doppler_plane_distance_velocity, dem_ecef, orbit_interpolator
    )

    if isinstance(orbit_time_guess, xr.DataArray):
        pass
    else:
        t_template = dem_ecef.isel({dim: 0}).drop_vars(dim).rename("azimuth_time")
        orbit_time_guess = xr.full_like(
            t_template,
            orbit_time_guess,
            dtype="float64",
        )

    if method == "secant":
        orbit_time_guess_prev = orbit_time_guess + orbit_time_prev_shift
        orbit_time, _, _, k, (dem_distance, satellite_velocity, sar_ecef) = secant_method(
            zero_doppler,
            orbit_time_guess_prev,
            orbit_time_guess,
            diff_ufunc,
            maxiter=maxiter,
        )
    elif method in {"newton", "newton_raphson"}:
        zero_doppler_prime = functools.partial(
            zero_doppler_plane_distance_velocity_prime, orbit_interpolator
        )
        orbit_time, _, k, (dem_distance, satellite_velocity, sar_ecef) = newton_raphson_method(
            zero_doppler,
            zero_doppler_prime,
            orbit_time_guess,
            diff_ufunc,
            maxiter=maxiter,
        )

    # sar_ecef = zero_doppler_plane_distance_velocity(dem_ecef, orbit_interpolator, orbit_time_guess)[1][2]
    ellipsoid_incidence_angle = None
    local_incidence_angle = None
    if calc_annotation:
        ellipsoid_incidence_angle = calculate_ellipsoid_incidence_angle_vectorised(
            sar_ecef=sar_ecef,
            dem_ecef=dem_ecef
        )
        local_incidence_angle = calculate_local_incidence_angle(dem_ecef, sar_ecef)
    return orbit_time, dem_distance, satellite_velocity, ellipsoid_incidence_angle, local_incidence_angle


def backward_geocode(
        dem_ecef: xr.DataArray,
        orbit_interpolator: orbit.OrbitPolyfitInterpolator,
        orbit_time_guess: xr.DataArray | float = 0.0,
        dim: str = "axis",
        zero_doppler_distance: float = 1.0,
        satellite_speed: float = 7_500.0,
        method: str = "newton",
        seed_step: tuple[int, int] | None = None,
        maxiter: int = 10,
        maxiter_after_seed: int = 1,
        orbit_time_prev_shift: float = -0.1,
        calc_annotation=False
) -> xr.Dataset:
    if seed_step is not None:
        dem_ecef_seed = dem_ecef.isel(
            y=slice(seed_step[0] // 2, None, seed_step[0]),
            x=slice(seed_step[1] // 2, None, seed_step[1]),
        )
        orbit_time_seed, _, _, _, _ = backward_geocode_simple(
            dem_ecef_seed,
            orbit_interpolator,
            orbit_time_guess,
            dim,
            zero_doppler_distance,
            satellite_speed,
            method,
            orbit_time_prev_shift=orbit_time_prev_shift,
        )
        orbit_time_guess = orbit_time_seed.interp_like(
            dem_ecef.sel(axis=0), kwargs={"fill_value": "extrapolate"}
        )
        maxiter = maxiter_after_seed

    orbit_time, dem_distance, satellite_velocity, ellipsoid_incidence_angle, local_incidence_angle = backward_geocode_simple(
        dem_ecef,
        orbit_interpolator,
        orbit_time_guess,
        dim,
        zero_doppler_distance,
        satellite_speed,
        method,
        maxiter=maxiter,
        orbit_time_prev_shift=orbit_time_prev_shift,
        calc_annotation=calc_annotation
    )

    acquisition = xr.Dataset(
        data_vars={
            "azimuth_time": orbit_interpolator.orbit_time_to_azimuth_time(orbit_time),
            "dem_distance": dem_distance,
            "satellite_velocity": satellite_velocity.transpose(*dem_distance.dims),
            "ellipsoid_incidence_angle": ellipsoid_incidence_angle,
            "local_incidence_angle": local_incidence_angle,
        }
    )
    return acquisition


def calculate_ellipsoid_incidence_angle(sar_ecef: xr.DataArray, dem_ecef: xr.DataArray):
    block_x = dem_ecef.sizes['x']
    block_y = dem_ecef.sizes['y']
    data = np.ndarray((block_y, block_x))
    data.fill(0.0)

    A = 6378137.0
    B = 6356752.3142451794975639665996337
    asq = A * A
    bsq = B * B

    arr = []
    for y in range(block_y):
        # uncomment for testing speedup
        # TODO this will be improved upon in the next commits to avoid loops in pure python
        # if y % 50 != 0:
        #    continue
        # print(y)

        for x in range(block_x):
            ecefx = dem_ecef[0].values[y][x]
            ecefy = dem_ecef[1].values[y][x]
            ecefz = dem_ecef[2].values[y][x]
            ep = np.array([ecefx, ecefy, ecefz])
            sar_pos = np.array(sar_ecef[y][x].values)

            ep_calc = np.array([ecefx / asq, ecefy / asq, ecefz / bsq])

            sar_dir = sar_pos - ep

            norm_ep = ep_calc / np.sqrt(np.sum(ep_calc ** 2))
            norm_sar = sar_dir / np.sqrt(np.sum(sar_dir ** 2))

            angle_rad = angle_between(norm_ep, norm_sar)
            deg = 360 * angle_rad / (math.pi * 2)
            data[y][x] = deg

    ellipsoid_incidence_angle = xr.DataArray(data=data)
    return ellipsoid_incidence_angle


def calculate_ellipsoid_incidence_angle_vectorised(sar_ecef: xr.DataArray, dem_ecef: xr.DataArray):
    A = 6378137.0
    B = 6356752.3142451794975639665996337
    asq = A * A
    bsq = B * B

    ellipsoid_scaler = np.array([1 / asq, 1 / asq, 1 / bsq]).reshape(3, 1, 1)
    ep_calc = dem_ecef * ellipsoid_scaler

    norm_ep = ep_calc / np.linalg.norm(ep_calc, axis=0)

    dem_ecef_transposed = dem_ecef.transpose("y", "x", dem_ecef.dims[0])
    sar_dir = sar_ecef - dem_ecef_transposed

    # Normalize the look direction vectors using NumPy's broadcasting
    norm_sar = sar_dir / np.linalg.norm(sar_dir.values, axis=2, keepdims=True)

    # 4. Calculate the angle between the two vector fields
    # Transpose norm_sar from (y, x, 3) back to (3, y, x) to align with norm_ep
    norm_sar_transposed = norm_sar.transpose(dem_ecef.dims[0], "y", "x")

    # Use einsum for an efficient, element-wise dot product across the grid
    dot_product = np.einsum("ijk,ijk->jk", norm_ep, norm_sar_transposed)

    # Calculate the angle, clipping for numerical stability, and convert to degrees
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.rad2deg(angle_rad)

    # 5. Return a properly formatted xarray DataArray
    spatial_dims = dem_ecef.dims[1:]
    return xr.DataArray(
        data=angle_deg.data,
        dims=spatial_dims,
        coords={dim: dem_ecef.coords[dim] for dim in spatial_dims},
        name="ellipsoid_incidence_angle",
    )


def calculate_dem_normals_ecef(dem_ecef: xr.DataArray) -> xr.DataArray:
    if dem_ecef.shape[0] != 3:
        # Heuristic: put the 3-length dim first
        comp_dim = [d for d in dem_ecef.dims if dem_ecef.sizes[d] == 3][0]
        dem_ecef_reshaped = dem_ecef.transpose(comp_dim, ...)
    else:
        dem_ecef_reshaped = dem_ecef

    gradient = np.gradient(dem_ecef_reshaped.values, axis=(1, 2))
    grad_y = gradient[0]
    grad_x = gradient[1]

    normals = np.cross(grad_x, grad_y, axis=0)

    norms = np.linalg.norm(normals, axis=0)
    norms[norms == 0] = 1
    normals /= norms

    p = dem_ecef_reshaped.values
    sign = np.sign(np.sum(normals * p, axis=0, keepdims=True))
    sign[sign == 0] = 1  # Avoid zero division
    normals *= sign

    return xr.DataArray(normals, dims=dem_ecef_reshaped.dims, coords=dem_ecef_reshaped.coords)


def calculate_local_incidence_angle(
        dem_ecef: xr.DataArray,
        sar_ecef: xr.DataArray
) -> xr.DataArray:
    def ensure_cyx(arr: xr.DataArray) -> xr.DataArray:
        if arr.shape[0] != 3:
            comp_dim = [d for d in arr.dims if arr.sizes[d] == 3][0]
            return arr.transpose(comp_dim, ...)
        return arr

    sar_ecef_reshaped = ensure_cyx(sar_ecef)
    dem_ecef_reshaped = ensure_cyx(dem_ecef)

    dem_normals = calculate_dem_normals_ecef(dem_ecef)
    los = sar_ecef_reshaped - dem_ecef_reshaped
    norm_los = los / np.linalg.norm(los, axis=0)

    dot_product = np.einsum('cyx,cyx->yx', dem_normals, norm_los)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.rad2deg(angle_rad)

    spatial_dims = dem_ecef.dims[1:]

    return xr.DataArray(
        data=angle_deg.data,
        dims=spatial_dims,
        coords={dim: dem_ecef.coords[dim] for dim in spatial_dims},
        name="local_incidence_angle",
    )
