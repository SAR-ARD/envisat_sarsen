"""Envisat product and utilities."""
from abc import ABC
from typing import Any, List, Dict

import numpy as np
import pandas as pd
import scipy.constants

import xarray as xr

import sarsen
from sarsen import datamodel


def make_orbit(azimuth_time: List[Any],
               positions: List[List[Any]],
               velocities: List[List[Any]],
               attrs: Dict[str, Any] = {},
               ) -> xr.DataArray:
    position = xr.Variable(data=positions, dims=("axis", "azimuth_time"))  # type: ignore
    velocity = xr.Variable(data=velocities, dims=("axis", "azimuth_time"))  # type: ignore

    ds = xr.DataArray(
        data=position,
        attrs=attrs,
        coords={
            "azimuth_time": [np.datetime64(dt, 'ns') for dt in azimuth_time],
            "axis": [0, 1, 2],
        },
    )
    # for data_var in ds.data_vars:
    #     ds[data_var].attrs = attrs

    return ds


class EnvisatProduct:
    measurement: xr.Dataset | None = None
    osv: xr.Dataset | None = None
    product_type: str = 'SLC'

    def __init__(self, path: str, *args: Any, **kwargs: Any) -> None:
        """
        Initialize an Envisat product.

        :param path: Path to the Envisat product file.
        """
        super().__init__(*args, **kwargs)
        self.path = path
        self.measurement = xr.open_dataset(path, engine='asar')
        self.osv = self.compute_osv()

    def beta_nought(self) -> xr.DataArray:
        return np.power(np.abs(self.measurement), 2)

    # def grid_parameters(
    #         self,
    #         grouping_area_factor: tuple[float, float] = (3.0, 3.0),
    # ) -> dict[str, Any]:
    #     return dict()

    def state_vectors(self):
        return self.osv

    # def interp_sar(
    #         self,
    #         *args: Any, **kwargs: Any
    # ) -> xr.DataArray:
    #     return sarsen.SlantRangeSarProduct.interp_sar(self, *args, **kwargs)

    def complex_amplitude(self) -> xr.DataArray:
        beta_nought = self.beta_nought()
        beta_nought = beta_nought.drop_vars(["pixel", "line"])
        return beta_nought

    def compute_osv(self):
        osv = self.measurement.attrs['metadata']["records"]["main_processing_params"]["orbit_state_vectors"]

        azimuth_times: List[Any] = []
        positions: List[List[Any]] = [[], [], []]
        velocities: List[List[Any]] = [[], [], []]
        for orbit in osv:
            azimuth_times.append(orbit["state_vect_time_1"])
            positions[0].append(orbit["x_pos_1"] * 1e-2)
            positions[1].append(orbit["y_pos_1"] * 1e-2)
            positions[2].append(orbit["z_pos_1"] * 1e-2)
            velocities[0].append(orbit["x_vel_1"] * 1e-5)
            velocities[1].append(orbit["y_vel_1"] * 1e-5)
            velocities[2].append(orbit["z_vel_1"] * 1e-5)

        return make_orbit(azimuth_times, positions, velocities)

