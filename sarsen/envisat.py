"""Envisat product and utilities."""
from abc import ABC
from typing import Any, List, Dict

import numpy as np
import datetime

import xarray as xr


def make_orbit(azimuth_time: List[Any],
               positions: List[List[Any]],
               velocities: List[List[Any]],
               attrs: Dict[str, Any] = {},
               ) -> xr.DataArray:
    """Constructs an xarray DataArray representing satellite orbit data.

    This function takes lists of timestamps, positions, and velocities and
    organizes them into a structured xarray DataArray with appropriate
    coordinates and dimensions.

    Args:
        azimuth_time (List[Any]): A list of timestamps for each state vector.
            Items will be converted to numpy.datetime64[ns].
        positions (List[List[Any]]): A list of 3D position vectors, where each
            inner list corresponds to a timestamp (e.g., [[x1, x2,...],
            [y1, y2,...], [z1, z2,...]]).
        velocities (List[List[Any]]): A list of 3D velocity vectors, structured
            identically to the positions.
        attrs (Dict[str, Any], optional): A dictionary of metadata to be
            attached to the resulting DataArray. Defaults to {}.

    Returns:
        xr.DataArray: A DataArray containing the orbit positions, with
        'azimuth_time' and 'axis' as coordinates. The velocity information
        is included for completeness but the primary data variable is position.
    """
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


def azimuth_slant_range_grid(
        attrs: dict[str, Any],
        grouping_area_factor: tuple[float, float] = (3.0, 3.0),
) -> dict[str, Any]:
    """Calculates grid spacing and timing parameters for the SAR product.

    This function determines the slant range and azimuth grid characteristics
    based on the product's metadata attributes. It adjusts the spacing based
    on the product type (SLC or other) and a grouping factor.

    Args:
        attrs (dict[str, Any]): The product's metadata attributes dictionary.
            Must contain keys like 'product_type', 'range_pixel_spacing', etc.
        grouping_area_factor (tuple[float, float], optional): Factors to scale
            the azimuth and range pixel spacing. Defaults to (3.0, 3.0).

    Returns:
        dict[str, Any]: A dictionary containing key grid parameters, including
        timing intervals, start times, and spacing in meters.
    """
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
    """Represents an ENVISAT ASAR satellite data product.

    This class provides a high-level interface to an ENVISAT product,
    encapsulating the measurement data, orbit state vectors (OSV), and metadata.

    Attributes:
        measurement (xr.Dataset | None): The main SAR measurement data, loaded
            as an xarray Dataset.
        osv (xr.Dataset | None): The Orbit State Vectors (OSV) associated with
            the product, represented as an xarray Dataset.
        product_type (str): The type of the ENVISAT product. Defaults to 'SLC'.
    """
    measurement: xr.Dataset | None = None
    osv: xr.Dataset | None = None
    product_type: str = 'SLC'

    def __init__(self, path: str, osv_file=None, *args: Any, **kwargs: Any) -> None:
        """Initializes and loads an Envisat product from a given path.

        Args:
            path (str): The file path to the Envisat product.
            osv_file (str | None, optional): The file path to an external
                DORVOR orbit file. If None, orbit data is read from the
                product's metadata. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.path = path
        self.measurement = xr.open_dataset(path, engine='asar')
        self.osv = self.compute_osv(osv_file)

    def beta_nought(self) -> xr.DataArray:
        """Calculates the Beta Nought backscatter coefficient.

        Beta Nought is a measure of the radar reflectivity of a target,
        normalized for the illuminated area. This method uses the calibration
        factor and vector from the product's metadata.

        Returns:
            xr.DataArray: The calibrated Beta Nought values as an xarray DataArray.
        """
        cal_factor = self.measurement.metadata["direct_parse"]["cal_factor"]
        cal_vector = self.measurement.metadata["direct_parse"]["cal_vector"]
        return (np.power(np.abs(self.measurement), 2) / cal_factor) * cal_vector

    def grid_parameters(
            self,
            grouping_area_factor: tuple[float, float] = (3.0, 3.0),
    ) -> dict[str, Any]:
        """Computes the azimuth and slant range grid parameters.

        This method generates grid information based on the product's metadata
        and a given grouping factor.

        Args:
            grouping_area_factor (tuple[float, float], optional): A factor to
                adjust the grouping area in azimuth and range.
                Defaults to (3.0, 3.0).

        Returns:
            dict[str, Any]: A dictionary containing the grid parameters.
        """
        return azimuth_slant_range_grid(self.measurement.attrs, grouping_area_factor)

    def state_vectors(self):
        """Retrieves the orbit state vectors (OSV).

        Returns:
            xr.Dataset | None: The orbit state vectors as an xarray Dataset,
            or None if not available.
        """
        return self.osv

    def complex_amplitude(self) -> xr.DataArray:
        """Computes the complex amplitude from the Beta Nought values.

        This method calculates the Beta Nought and then removes the 'pixel'
        and 'line' coordinates to return a clean complex amplitude array.

        Returns:
            xr.DataArray: The complex amplitude as an xarray DataArray.
        """
        beta_nought = self.beta_nought()
        beta_nought = beta_nought.drop_vars(["pixel", "line"])
        return beta_nought

    def compute_osv(self, osv_file=None):
        """Computes the orbit state vectors for the product.

        This method can either parse an external ENVISAT DORVOR orbit file or
        extract the state vectors directly from the product's metadata.

        Args:
            osv_file (str | None, optional): Path to an external DORVOR file.
                If provided, orbit data will be parsed from this file. If None,
                orbit data is extracted from the product's own metadata records.
                Defaults to None.

        Returns:
            xr.Dataset: An xarray Dataset containing the computed orbit state
            vectors, including time, position, and velocity.

        Raises:
            RuntimeError: If an `osv_file` is provided but contains fewer than
                6 valid OSV points within the required time window.
        """
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
