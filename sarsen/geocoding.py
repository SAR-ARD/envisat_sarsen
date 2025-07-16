"""Reference "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019.

See: https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3
"""

import functools
from typing import Any, Callable, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import xarray as xr

import math

import rasterio

TimedeltaArrayLike = TypeVar("TimedeltaArrayLike", bound=npt.ArrayLike)
FloatArrayLike = TypeVar("FloatArrayLike", bound=npt.ArrayLike)


import numpy as np


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))




def secant_method(
        ufunc: Callable[[TimedeltaArrayLike], Tuple[FloatArrayLike, FloatArrayLike]],
        t_prev: TimedeltaArrayLike,
        t_curr: TimedeltaArrayLike,
        diff_ufunc: float = 1.0,
        diff_t: np.timedelta64 = np.timedelta64(0, "ns"),
) -> Tuple[TimedeltaArrayLike, TimedeltaArrayLike, FloatArrayLike, Any]:
    """Return the root of ufunc calculated using the secant method."""
    # implementation modified from https://en.wikipedia.org/wiki/Secant_method
    f_prev, _ = ufunc(t_prev)

    # strong convergence, all points below one of the two thresholds
    while True:
        f_curr, payload_curr = ufunc(t_curr)

        # the `not np.any` construct let us accept `np.nan` as good values
        if not np.any((np.abs(f_curr) > diff_ufunc)):
            break

        t_diff: TimedeltaArrayLike
        p: TimedeltaArrayLike
        q: FloatArrayLike

        t_diff = t_curr - t_prev  # type: ignore
        p = f_curr * t_diff  # type: ignore
        q = f_curr - f_prev  # type: ignore

        # t_prev, t_curr = t_curr, t_curr - f_curr * np.timedelta64(-148_000, "ns")
        t_prev, t_curr = t_curr, t_curr - np.where(q != 0, p / q, 0)  # type: ignore
        f_prev = f_curr

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

    return t_curr, t_prev, f_curr, payload_curr


# FIXME: interpolationg the direction decreses the precision, this function should
#   probably have velocity_ecef_sar in input instead
def zero_doppler_plane_distance(
        dem_ecef: xr.DataArray,
        position_ecef_sar: xr.DataArray,
        direction_ecef_sar: xr.DataArray,
        azimuth_time: TimedeltaArrayLike,
        dim: str = "axis",
) -> Tuple[xr.DataArray, Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    sar_ecef = position_ecef_sar.interp(azimuth_time=azimuth_time)
    dem_distance = dem_ecef - position_ecef_sar.interp(azimuth_time=azimuth_time)
    satellite_direction = direction_ecef_sar.interp(azimuth_time=azimuth_time)
    plane_distance = (dem_distance * satellite_direction).sum(dim, skipna=False)
    return plane_distance, (dem_distance, satellite_direction, sar_ecef)


def backward_geocode(
        dem_ecef: xr.DataArray,
        position_ecef: xr.DataArray,
        velocity_ecef: xr.DataArray,
        azimuth_time: Optional[xr.DataArray] = None,
        dim: str = "axis",
        diff_ufunc: float = 1.0,
) -> xr.Dataset:
    direction_ecef = (
            velocity_ecef / xr.dot(velocity_ecef, velocity_ecef, dim=dim) ** 0.5
    )

    zero_doppler = functools.partial(
        zero_doppler_plane_distance, dem_ecef, position_ecef, direction_ecef
    )

    if azimuth_time is None:
        azimuth_time = position_ecef.azimuth_time
    t_template = dem_ecef.isel({dim: 0}).drop_vars(dim)
    t_prev = xr.full_like(t_template, azimuth_time.values[0], dtype=azimuth_time.dtype)
    t_curr = xr.full_like(t_template, azimuth_time.values[-1], dtype=azimuth_time.dtype)

    # NOTE: dem_distance has the associated azimuth_time as a coordinate already
    _, _, _, (dem_distance, satellite_direction, sar_ecef) = secant_method(
        zero_doppler,
        t_prev,
        t_curr,
        diff_ufunc,
    )

    pos_ecef_calc = position_ecef.interp(azimuth_time=t_curr)




    # pure python ellipsoid calculation to generate a new annotation layer 

    block_x = dem_ecef.sizes["x"]
    block_y = dem_ecef.sizes["y"]
    data = np.ndarray((block_y, block_x))
    data.fill(0.0)

    #
    A = 6378137.0
    B = 6356752.3142451794975639665996337
    asq = A * A
    bsq = B * B

    arr = []
    for y in range(block_y):
        # uncomment for testing speedup
        # TODO this will be improved upon in the next commits to avoid loops in pure python
        #if y % 50 != 0:
        #    continue
        #print(y)

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

    acquisition = xr.Dataset(
        data_vars={
            "dem_distance": dem_distance,
            "satellite_direction": satellite_direction.transpose(*dem_distance.dims),
            "ellipsoid_incidence_angle": ellipsoid_incidence_angle
        }
    )
    return acquisition.reset_coords("azimuth_time")
