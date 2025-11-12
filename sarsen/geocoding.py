"""Reference "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019.

See: https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3
"""

import functools
from typing import Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt
import xarray as xr

from sarsen import orbit

ArrayLike = TypeVar("ArrayLike", bound=npt.ArrayLike)

FloatArrayLike = TypeVar("FloatArrayLike", bound=npt.ArrayLike)


def unit_vector(
    vector: np.ndarray[Any, np.dtype[np.floating]],
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """
    Return the unit vector of the input vector.

    Parameters
    ----------
    vector : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Unit vector in the same direction as the input.
    """
    return vector / np.linalg.norm(vector)


def angle_between(
    v1: np.ndarray[Any, np.dtype[np.floating]],
    v2: np.ndarray[Any, np.dtype[np.floating]],
) -> float:
    """
    Calculate the angle in radians between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First input vector.
    v2 : np.ndarray
        Second input vector.

    Returns
    -------
    float
        Angle in radians between the two vectors.
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return float(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


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
        - satellite_velocity**2
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
    calc_annotation: bool = False,
) -> tuple[
    xr.DataArray,  # orbit_time
    xr.DataArray,  # dem_distance
    xr.DataArray,  # satellite_velocity
    xr.DataArray | None,  # ellipsoid_incidence_angle
    xr.DataArray | None,  # local_incidence_angle
    xr.DataArray | None,  # layover_shadow_mask
    xr.DataArray | None,  # gamma_sigma_ratio
]:
    """
    Perform a single backward geocoding step to estimate the satellite azimuth time for each DEM pixel.

    This function solves for the orbit time (azimuth time) at which the satellite is closest to each DEM pixel
    using either the secant or Newton-Raphson method. Optionally, it can also compute annotation
    layers such as incidence angles, layover/shadow mask, and gamma/sigma ratio.

    Parameters
    ----------
    dem_ecef : xr.DataArray
        DEM points in ECEF coordinates, shape (3, ...).
    orbit_interpolator : orbit.OrbitPolyfitInterpolator
        Interpolator for satellite orbit position and velocity.
    orbit_time_guess : xr.DataArray or float, optional
        Initial guess for the orbit time (default: 0.0).
    dim : str, optional
        Name of the coordinate axis dimension (default: "axis").
    zero_doppler_distance : float, optional
        Tolerance for the zero-Doppler plane distance (default: 1.0).
    satellite_speed : float, optional
        Nominal satellite speed in m/s (default: 7500.0).
    method : str, optional
        Root-finding method: "secant" or "newton" (default: "secant").
    orbit_time_prev_shift : float, optional
        Shift for the previous orbit time guess in the secant method (default: -0.1).
    maxiter : int, optional
        Maximum number of iterations for the root-finding method (default: 10).
    calc_annotation : bool, optional
        If True, calculate annotation layers (default: False).

    Returns
    -------
    tuple
        Tuple containing:
        - orbit_time: xr.DataArray
        - dem_distance: xr.DataArray
        - satellite_velocity: xr.DataArray
        - ellipsoid_incidence_angle: xr.DataArray or None
        - local_incidence_angle: xr.DataArray or None
        - layover_shadow_mask: xr.DataArray or None
        - gamma_sigma_ratio: xr.DataArray or None
    """
    diff_ufunc: float = zero_doppler_distance * satellite_speed
    zero_doppler: Callable[
        [xr.DataArray],
        tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray, xr.DataArray]],
    ] = functools.partial(
        zero_doppler_plane_distance_velocity, dem_ecef, orbit_interpolator
    )

    if isinstance(orbit_time_guess, xr.DataArray):
        pass
    else:
        t_template: xr.DataArray = (
            dem_ecef.isel({dim: 0}).drop_vars(dim).rename("azimuth_time")
        )
        orbit_time_guess = xr.full_like(
            t_template,
            orbit_time_guess,
            dtype="float64",
        )

    if method == "secant":
        orbit_time_guess_prev: xr.DataArray = orbit_time_guess + orbit_time_prev_shift
        orbit_time, _, _, k, (dem_distance, satellite_velocity, sar_ecef) = (
            secant_method(
                zero_doppler,
                orbit_time_guess_prev,
                orbit_time_guess,
                diff_ufunc,
                maxiter=maxiter,
            )
        )
    elif method in {"newton", "newton_raphson"}:
        zero_doppler_prime: Callable[
            [xr.DataArray, tuple[xr.DataArray, xr.DataArray, xr.DataArray]],
            xr.DataArray,
        ] = functools.partial(
            zero_doppler_plane_distance_velocity_prime, orbit_interpolator
        )
        orbit_time, _, k, (dem_distance, satellite_velocity, sar_ecef) = (
            newton_raphson_method(
                zero_doppler,
                zero_doppler_prime,
                orbit_time_guess,
                diff_ufunc,
                maxiter=maxiter,
            )
        )

    ellipsoid_incidence_angle: xr.DataArray | None = None
    local_incidence_angle: xr.DataArray | None = None
    layover_shadow_mask: xr.DataArray | None = None
    gamma_sigma_ratio: xr.DataArray | None = None
    if calc_annotation:
        ellipsoid_incidence_angle = calculate_ellipsoid_incidence_angle(
            sar_ecef=sar_ecef, dem_ecef=dem_ecef
        )
        local_incidence_angle = calculate_local_incidence_angle(dem_ecef, sar_ecef)
        layover_shadow_mask = calculate_layover_shadow_mask(
            local_incidence_angle, ellipsoid_incidence_angle
        )
        gamma_sigma_ratio = calculate_gamma_sigma_ratio(local_incidence_angle)
    return (
        orbit_time,
        dem_distance,
        satellite_velocity,
        ellipsoid_incidence_angle,
        local_incidence_angle,
        layover_shadow_mask,
        gamma_sigma_ratio,
    )


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
    calc_annotation: bool = False,
) -> xr.Dataset:
    """
    Perform backward geocoding to estimate satellite azimuth time and annotation layers for each DEM pixel.

    Parameters
    ----------
    dem_ecef : xr.DataArray
        DEM points in ECEF coordinates, shape (3, ...).
    orbit_interpolator : orbit.OrbitPolyfitInterpolator
        Interpolator for satellite orbit position and velocity.
    orbit_time_guess : xr.DataArray or float, optional
        Initial guess for the orbit time (default: 0.0).
    dim : str, optional
        Name of the coordinate axis dimension (default: "axis").
    zero_doppler_distance : float, optional
        Tolerance for the zero-Doppler plane distance (default: 1.0).
    satellite_speed : float, optional
        Nominal satellite speed in m/s (default: 7500.0).
    method : str, optional
        Root-finding method: "secant" or "newton" (default: "newton").
    seed_step : tuple[int, int] or None, optional
        Step size for seeding the initial guess (default: None).
    maxiter : int, optional
        Maximum number of iterations for the root-finding method (default: 10).
    maxiter_after_seed : int, optional
        Maximum number of iterations after seeding (default: 1).
    orbit_time_prev_shift : float, optional
        Shift for the previous orbit time guess in the secant method (default: -0.1).
    calc_annotation : bool, optional
        If True, calculate annotation layers (default: False).

    Returns
    -------
    xr.Dataset
        Dataset containing azimuth time, DEM distance, satellite velocity, and optional annotation layers.
    """
    if seed_step is not None:
        dem_ecef_seed = dem_ecef.isel(
            y=slice(seed_step[0] // 2, None, seed_step[0]),
            x=slice(seed_step[1] // 2, None, seed_step[1]),
        )
        orbit_time_seed, _, _, _, _, _, _ = backward_geocode_simple(
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

    (
        orbit_time,
        dem_distance,
        satellite_velocity,
        ellipsoid_incidence_angle,
        local_incidence_angle,
        layover_shadow_mask,
        gamma_sigma_ratio,
    ) = backward_geocode_simple(
        dem_ecef,
        orbit_interpolator,
        orbit_time_guess,
        dim,
        zero_doppler_distance,
        satellite_speed,
        method,
        maxiter=maxiter,
        orbit_time_prev_shift=orbit_time_prev_shift,
        calc_annotation=calc_annotation,
    )

    acquisition = xr.Dataset(
        data_vars={
            "azimuth_time": orbit_interpolator.orbit_time_to_azimuth_time(orbit_time),
            "dem_distance": dem_distance,
            "satellite_velocity": satellite_velocity.transpose(*dem_distance.dims),
            "ellipsoid_incidence_angle": ellipsoid_incidence_angle,
            "local_incidence_angle": local_incidence_angle,
            "layover_shadow_mask": layover_shadow_mask,
            "gamma_sigma_ratio": gamma_sigma_ratio,
        }
    )
    return acquisition


def calculate_ellipsoid_incidence_angle(
    sar_ecef: xr.DataArray, dem_ecef: xr.DataArray
) -> xr.DataArray:
    """
    Calculate the ellipsoid incidence angle for each DEM pixel.

    The ellipsoid incidence angle is the angle between the local ellipsoid normal
    at the DEM point and the line-of-sight vector from the satellite to the DEM point.

    Parameters
    ----------
    sar_ecef : xr.DataArray
        Satellite position(s) in ECEF coordinates, shape (3, y, x).
    dem_ecef : xr.DataArray
        DEM points in ECEF coordinates, shape (3, y, x).

    Returns
    -------
    xr.DataArray
        Ellipsoid incidence angle in degrees, with the same spatial dimensions as the input.
    """
    A = 6378137.0
    B = 6356752.3142451794975639665996337
    asq = A * A
    bsq = B * B

    ellipsoid_scaler = np.array([1 / asq, 1 / asq, 1 / bsq]).reshape(3, 1, 1)
    ep_calc = dem_ecef * ellipsoid_scaler

    norm_ep = ep_calc / np.linalg.norm(ep_calc, axis=0)

    dem_ecef_transposed = dem_ecef.transpose("y", "x", dem_ecef.dims[0])
    sar_dir = sar_ecef - dem_ecef_transposed

    norm_sar = sar_dir / np.linalg.norm(sar_dir.values, axis=2, keepdims=True)

    norm_sar_transposed = norm_sar.transpose(dem_ecef.dims[0], "y", "x")

    dot_product = np.einsum("ijk,ijk->jk", norm_ep, norm_sar_transposed)

    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.rad2deg(angle_rad)

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

    return xr.DataArray(
        normals, dims=dem_ecef_reshaped.dims, coords=dem_ecef_reshaped.coords
    )


def calculate_local_incidence_angle(
    dem_ecef: xr.DataArray, sar_ecef: xr.DataArray
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

    dot_product = np.einsum("cyx,cyx->yx", dem_normals, norm_los)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.rad2deg(angle_rad)

    spatial_dims = dem_ecef.dims[1:]

    return xr.DataArray(
        data=angle_deg.data,
        dims=spatial_dims,
        coords={dim: dem_ecef.coords[dim] for dim in spatial_dims},
        name="local_incidence_angle",
    )


def calculate_gamma_sigma_ratio(local_incidence_angle: xr.DataArray) -> xr.DataArray:
    """
    Calculate the gamma/sigma ratio for geometric correction in SAR geocoding.

    The ratio is defined as 1 / cos(local_incidence_angle), where the local incidence angle
    is given in degrees. This factor is used to convert between gamma nought and sigma nought
    backscatter coefficients.

    Parameters
    ----------
    local_incidence_angle : xr.DataArray
        The local incidence angle in degrees.

    Returns
    -------
    xr.DataArray
        The gamma/sigma ratio as a DataArray, with the same dimensions and coordinates as the input.
    """
    local_inc_data = local_incidence_angle.data
    local_inc_rad = np.deg2rad(local_inc_data)

    ratio_dask = 1.0 / np.cos(local_inc_rad)

    gamma_sigma_ratio = xr.DataArray(
        data=ratio_dask,
        dims=local_incidence_angle.dims,
        coords=local_incidence_angle.coords,
        name="gamma_sigma_ratio",
        attrs={
            "long_name": "gamma_nought_sigma_nought_ratio",
            "description": "Geometric correction factor: 1 / cos(local_incidence_angle)",
            "units": "unitless",
        },
    )
    return gamma_sigma_ratio


def calculate_layover_shadow_mask(
    local_incidence_angle: xr.DataArray | None,
    ellipsoid_incidence_angle: xr.DataArray | None,
) -> xr.DataArray:
    """
    Calculate the layover and shadow mask for SAR geocoding.

    Coding:
        0: not layover, not shadow
        1: layover
        2: shadow
        3: layover in shadow

    A pixel is marked as:
        - shadow if its local incidence angle is greater than 90 degrees,
        - layover if its local incidence angle is less than the ellipsoid incidence angle.

    Parameters
    ----------
    local_incidence_angle : xr.DataArray
        The local incidence angle in degrees.
    ellipsoid_incidence_angle : xr.DataArray
        The ellipsoid incidence angle in degrees.

    Returns
    -------
    xr.DataArray
        Integer mask with the same shape as the input angles.
    """
    if local_incidence_angle is None or ellipsoid_incidence_angle is None:
        raise ValueError(
            "Both local_incidence_angle and ellipsoid_incidence_angle must be provided."
        )
    local_angle_data = local_incidence_angle.data
    ellipsoid_angle_data = ellipsoid_incidence_angle.data

    shadow_mask = local_angle_data > 90.0
    layover_mask = local_angle_data < ellipsoid_angle_data

    combined_mask = layover_mask.astype(np.uint8) + 2 * shadow_mask.astype(np.uint8)

    return xr.DataArray(
        data=combined_mask,
        dims=local_incidence_angle.dims,
        coords=local_incidence_angle.coords,
        name="Layover-Shadow mask",
        attrs={
            "flag_values": np.array([0, 1, 2, 3], dtype=np.uint8),
            "flag_meanings": "not_layover_not_shadow layover shadow layover_in_shadow",
        },
    )
