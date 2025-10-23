"""Envisat product and utilities."""

import datetime
import logging
from typing import Any, Dict, List, cast

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def make_orbit(
    azimuth_time: List[Any],
    positions: List[List[Any]],
    velocities: List[List[Any]],
    attrs: Dict[str, Any] = {},
) -> xr.DataArray:
    position = xr.Variable(data=positions, dims=("axis", "azimuth_time"))

    ds = xr.DataArray(
        data=position,
        attrs=attrs,
        coords={
            "azimuth_time": [np.datetime64(dt, "ns") for dt in azimuth_time],
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
    def __init__(
        self,
        path: str,
        osv_file: str | None = None,
        polarization: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an Envisat product.

        :param path: Path to the Envisat product file.
        :param osv_file DORIS orbit file for Envisat, REAPER Version 2 DEOS orbits for ERS1/2
        :param polarization APS/APP mode polarization
        """
        super().__init__(*args, **kwargs)

        self.path = path
        self.measurement = xr.open_dataset(
            path, engine="asar", polarization=polarization
        )
        self.product_type = self.measurement.product_type
        self.osv = self.compute_osv(osv_file)

    def beta_nought(self) -> xr.DataArray:
        cal_factor = self.measurement.metadata["direct_parse"]["cal_factor"]
        cal_vector = self.measurement.metadata["direct_parse"]["cal_vector"]
        result = (np.power(np.abs(self.measurement), 2) / cal_factor) * cal_vector
        return cast(xr.DataArray, result)

    def grid_parameters(
        self,
        grouping_area_factor: tuple[float, float] = (3.0, 3.0),
    ) -> dict[str, Any]:
        return azimuth_slant_range_grid(self.measurement.attrs, grouping_area_factor)

    def state_vectors(self) -> xr.DataArray:
        return self.osv

    def complex_amplitude(self) -> xr.DataArray:
        beta_nought = self.beta_nought()
        beta_nought = beta_nought.drop_vars(["pixel", "line"])
        return beta_nought

    def compute_osv(self, osv_file: str | None = None) -> xr.DataArray:
        azimuth_times: List[Any] = []
        positions: List[List[Any]] = [[], [], []]
        velocities: List[List[Any]] = [[], [], []]

        if osv_file:
            # osv file passed as argument, parse a DORIS Envisat orbit file

            first_line_time = self.measurement.attrs["metadata"]["first_line_time"]
            first_line_dt = datetime.datetime.strptime(
                str(first_line_time)[:-3], "%Y-%m-%dT%H:%M:%S.%f"
            )

            osv_time_delta = 8

            start_time = first_line_dt + datetime.timedelta(minutes=-osv_time_delta)
            end_time = first_line_dt + datetime.timedelta(minutes=osv_time_delta)
            product_name = self.measurement.metadata["product"]
            if ".N1" in product_name:
                # parse a DORVOR orbit file for envisat
                orbit_content = None
                with open(osv_file, "r") as fp:
                    orbit_content = fp.read()

                orbit_content = orbit_content[1625:]
                orbit_lines = orbit_content.split("\n")

                for line in orbit_lines:
                    elems = line.split(" ")
                    if len(elems) < 10:
                        continue
                    orbit_dt = elems[0] + " " + elems[1]
                    osv_timestamp = datetime.datetime.strptime(
                        orbit_dt, "%d-%b-%Y %H:%M:%S.%f"
                    )

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
            elif ".E1" in product_name or ".E2" in product_name:
                # parse a ERS POD REAPER orbit file
                lines = None
                with open(osv_file, "r") as fp:
                    lines = fp.readlines()

                if ".E2" in product_name and not "L32" in lines[2]:
                    raise RuntimeError("ERS2 product and orbit file does not have the ERS1 tag(L32)")
                if ".E1" in product_name and not "L31" in lines[2]:
                    raise RuntimeError("ERS1 product and orbit file does not have the ERS1 tag(L31)")

                n_lines = len(lines)
                i = 0

                # from "Specifications for POD Products for ERS Altimetry Reprocessing"
                # Timestamps are in GPS time
                # TAI = GPS + 19s
                # https://data.iana.org/time-zones/data/leap-seconds.list
                # UTC = TAI - DTAI
                """
                DTAI
                26      # 1 Jan 1991
                27      # 1 Jul 1992
                28      # 1 Jul 1993
                29      # 1 Jul 1994
                30      # 1 Jan 1996
                31      # 1 Jul 1997
                32      # 1 Jan 1999
                33      # 1 Jan 2006
                34      # 1 Jan 2009
                35      # 1 Jul 2012
                """
                jan91 = datetime.datetime(1991, 1, 1)
                jul91 = datetime.datetime(1992, 7, 1)
                jul93 = datetime.datetime(1993, 7, 1)
                jul94 = datetime.datetime(1994, 7, 1)
                jan96 = datetime.datetime(1996, 1, 1)
                jul97 = datetime.datetime(1997, 7, 1)
                jan99 = datetime.datetime(1999, 1, 1)
                jan06 = datetime.datetime(2006, 1, 1)
                jan09 = datetime.datetime(2009, 1, 1)
                jul12 = datetime.datetime(2012, 7, 1)
                leap_dates = [jan91, jul91, jul93, jul94, jan96, jul97, jan99, jan06, jan09, jul12]
                leap_offsets = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
                leap_idx = -1
                for i in range(len(leap_dates) - 1):
                    if first_line_dt >= leap_dates[i] and first_line_dt <= leap_dates[i + 1]:
                        leap_idx = i
                        break
                if leap_idx == -1:
                    raise RunetimeError("Product start time = {} outside of know ERS1-2 lifetime".format(first_line_dt))

                leap_second = 19 - leap_offsets[leap_idx]

                while i < n_lines - 3:

                    if lines[i].startswith("*  ") and lines[i + 1].startswith('PL3') and lines[i + 2].startswith(
                            'VL3'):

                        time_tokens = ' '.join(lines[i].split()).split(" ")
                        time_tokens = time_tokens[1:7]
                        time_tokens = [float(x) for x in time_tokens]
                        time_tokens = [int(x) for x in time_tokens]

                        pos_tokens = ' '.join(lines[i + 1].split()).split(" ")
                        pos_tokens = pos_tokens[1:4]
                        pos_xyz = [float(x) for x in pos_tokens]
                        pos_xyz = [x * 1000.0 for x in pos_xyz]
                        vel_tokens = ' '.join(lines[i + 2].split()).split(" ")

                        vel_tokens = vel_tokens[1:4]
                        vel_xyz = [float(i) for i in vel_tokens]
                        osv_timestamp = datetime.datetime(time_tokens[0], time_tokens[1], time_tokens[2],
                                                          time_tokens[3],
                                                          time_tokens[4], time_tokens[5])

                        osv_timestamp += datetime.timedelta(seconds=leap_second)

                        if osv_timestamp >= start_time and osv_timestamp <= end_time:
                            azimuth_times.append(np.datetime64(osv_timestamp))
                            positions[0].append(pos_xyz[0])
                            positions[1].append(pos_xyz[1])
                            positions[2].append(pos_xyz[2])

                            velocities[0].append(vel_xyz[0])
                            velocities[1].append(vel_xyz[1])
                            velocities[2].append(vel_xyz[2])
                        i += 3
                    else:
                        i += 1

            else:
                raise RuntimeError(
                    "Unknown product, the product name({}) must contain .N1, .E1 or .E2\n".format(product_name))

        else:
            logger.warning(
                "Using ENVISAT format internal orbit data, "
                "this produces unstable results and is only for ease of testing"
            )
            osv = self.measurement.attrs["metadata"]["records"][
                "main_processing_params"
            ]["orbit_state_vectors"]

            for orbit in osv:
                azimuth_times.append(orbit["state_vect_time_1"])
                positions[0].append(orbit["x_pos_1"] * 1e-2)
                positions[1].append(orbit["y_pos_1"] * 1e-2)
                positions[2].append(orbit["z_pos_1"] * 1e-2)
                velocities[0].append(orbit["x_vel_1"] * 1e-5)
                velocities[1].append(orbit["y_vel_1"] * 1e-5)
                velocities[2].append(orbit["z_vel_1"] * 1e-5)

        return make_orbit(azimuth_times, positions, velocities)
