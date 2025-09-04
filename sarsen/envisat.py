"""Envisat product and utilities."""
from abc import ABC
from typing import Any, List, Dict

import numpy as np
import pandas as pd
import scipy.constants
import datetime

import xarray as xr


def make_orbit(azimuth_time: List[Any],
               positions: List[List[Any]],
               velocities: List[List[Any]],
               attrs: Dict[str, Any] = {},
               ) -> xr.DataArray:
    position = xr.Variable(data=positions, dims=("axis", "azimuth_time"))  # type: ignore

    ds = xr.DataArray(
        data=position,
        attrs=attrs,
        coords={
            "azimuth_time": [np.datetime64(dt, 'ns') for dt in azimuth_time],
            "axis": [0, 1, 2],
        },
    )

    return ds


def azimuth_slant_range_grid(
        attrs: dict[str, Any],
        grouping_area_factor: tuple[float, float] = (3.0, 3.0),
) -> dict[str, Any]:
    if attrs["product_type"] == "SLC":
        slant_range_spacing_m = (
                attrs["range_pixel_spacing"]
                * np.sin(attrs["incidence_angle_mid_swath"])
                * grouping_area_factor[1]
        )
    else:
        slant_range_spacing_m = attrs["range_pixel_spacing"] * grouping_area_factor[1]

    c = 299792458
    slant_range_time_interval_s = (
            slant_range_spacing_m * 2 / c  # ignore type
    )

    grid_parameters: dict[str, Any] = {
        "slant_range_time0": attrs["image_slant_range_time"],
        "slant_range_time_interval_s": slant_range_time_interval_s,
        "slant_range_spacing_m": slant_range_spacing_m,
        "azimuth_time0": np.datetime64(attrs["product_first_line_utc_time"]),
        "azimuth_time_interval_s": attrs["azimuth_time_interval"]
                                   * grouping_area_factor[0],
        "azimuth_spacing_m": attrs["azimuth_pixel_spacing"] * grouping_area_factor[0],
    }
    return grid_parameters


class EnvisatProduct:

    def __init__(self, path: str, osv_file=None, *args: Any, **kwargs: Any) -> None:
        """
        Initialize an Envisat product.

        :param path: Path to the Envisat product file.
        """
        super().__init__(*args, **kwargs)
        self.path = path
        self.measurement = xr.open_dataset(path, engine='asar')
        self.product_type = self.measurement.product_type
        self.osv = self.compute_osv(osv_file)

    def beta_nought(self) -> xr.DataArray:
        cal_factor = self.measurement.metadata["direct_parse"]["cal_factor"]
        cal_vector = self.measurement.metadata["direct_parse"]["cal_vector"]
        return (np.power(np.abs(self.measurement), 2) / cal_factor) * cal_vector

    def grid_parameters(
            self,
            grouping_area_factor: tuple[float, float] = (3.0, 3.0),
    ) -> dict[str, Any]:
        return azimuth_slant_range_grid(self.measurement.attrs, grouping_area_factor)

    def state_vectors(self):
        return self.osv

    def complex_amplitude(self) -> xr.DataArray:
        beta_nought = self.beta_nought()
        beta_nought = beta_nought.drop_vars(["pixel", "line"])
        return beta_nought

    def compute_osv(self, osv_file=None):

        azimuth_times: List[Any] = []
        positions: List[List[Any]] = [[], [], []]
        velocities: List[List[Any]] = [[], [], []]

        if osv_file:
            # osv file passed as argument, parse a DORVOR Envisat orbit file
            orbit_content = None
            with open(osv_file, "r") as fp:
                orbit_content = fp.read()

            orbit_content = orbit_content[1625:]
            orbit_lines = orbit_content.split("\n")

            first_line_time = self.measurement.attrs["metadata"]["first_line_time"]
            first_line_dt = datetime.datetime.strptime(str(first_line_time)[:-3], '%Y-%m-%dT%H:%M:%S.%f')

            osv_time_delta = 8

            start_time = first_line_dt + datetime.timedelta(minutes=-osv_time_delta)
            end_time = first_line_dt + datetime.timedelta(minutes=osv_time_delta)
            for line in orbit_lines:
                elems = line.split(" ")
                if len(elems) < 10:
                    continue
                orbit_dt = elems[0] + " " + elems[1]
                osv_timestamp = datetime.datetime.strptime(orbit_dt, "%d-%b-%Y %H:%M:%S.%f")

                if osv_timestamp < start_time or osv_timestamp > end_time:
                    continue

                pos_x = float(elems[4])
                pos_y = float(elems[5])
                pos_z = float(elems[6])

                vel_x = float(elems[7])
                vel_y = float(elems[8])
                vel_z = float(elems[9])

                azimuth_times.append(np.datetime64(osv_timestamp))
                positions[0].append(pos_x)
                positions[1].append(pos_y)
                positions[2].append(pos_z)

                velocities[0].append(vel_x)
                velocities[1].append(vel_y)
                velocities[2].append(vel_z)

            if len(azimuth_times) <= 5:
                raise RuntimeError("Not enough OSV points parsed from {}\n", osv_file)

        else:

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
