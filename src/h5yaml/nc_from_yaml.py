#
# This file is part of Python package: `h5yaml`
#
#     https://github.com/rmvanhees/pyxarr.git
#
# Copyright (c) 2025-2026 - R.M. van Hees (SRON)
#    All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Initialise netCDF4 file from a YAML configuration file using `h5py` or `netCDF4`."""

from __future__ import annotations

__all__ = ["NcFromYaml"]

# import logging
import pprint
from pathlib import Path

import yaml
import yaml_include

from .h5_create import H5Create
from .nc_create import NcCreate


# - local function -------------------------------------
def _from_yaml(file_path: Path | str) -> dict:
    """Read settings from a YAML file: `file_path`.

    Parameters
    ----------
    file_path :  Path | str
       full path of YAML file

    Returns
    -------
    dict
       content of the configuration file

    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path} not found")

    # Register the !include tag
    yaml.add_constructor(
        "!inc", yaml_include.Constructor(base_dir=file_path.parent), yaml.SafeLoader
    )

    with file_path.open("r", encoding="ascii") as fid:
        try:
            settings = yaml.safe_load(fid)
        except yaml.parser.ParserError as exc:
            raise RuntimeError(f"Failed to parse {file_path}") from exc

    return settings


# - class definition -----------------------------------
class NcFromYaml(H5Create):
    """Class to create a netCDF4 formated file from a YAML configuration file.

    Parameters
    ----------
    nc_yaml_fl :  Path | str | list[Path | str]
       YAML files with the HDF5 format definition

    """

    def __init__(
        self: NcFromYaml,
        nc_yaml_fl: Path | str | list[Path | str],
    ) -> None:
        """Construct a NcFromYaml instance."""
        # self.logger = logging.getLogger("h5yaml.NcFromYaml")
        self.module = "h5py"
        super().__init__()

        for yaml_fl in nc_yaml_fl if isinstance(nc_yaml_fl, list) else [nc_yaml_fl]:
            try:
                config = _from_yaml(yaml_fl)
            except RuntimeError as exc:
                raise RuntimeError from exc

            if "groups" in config:
                self.groups |= set(config["groups"])
            if "compounds" in config:
                self.compounds |= config["compounds"]
            if "dimensions" in config:
                self.dimensions |= config["dimensions"]
            if "variables" in config:
                self.variables |= config["variables"]
            if "attrs_global" in config:
                self.attrs_global |= config["attrs_global"]
            if "attrs_groups" in config:
                self.attrs_groups |= config["attrs_groups"]

    def __repr__(self: NcFromYaml) -> str:
        """Show object as dictionary."""
        return pprint.pformat(self.asdict)

    @property
    def asdict(self: NcFromYaml) -> dict:
        """Return definition of the HDF5/netCDF4 product."""
        return {
            "groups": self.groups,
            "dimensions": self.dimensions,
            "compounds": self.compounds,
            "variables": self.variables,
            "attrs_global": self.attrs_global,
            "attrs_groups": self.attrs_groups,
        }

    def use_netcdf4(self: NcFromYaml) -> NcFromYaml:
        """Use module netCDF4 to generate the HDF5/netCDF4 file."""
        if self.module == "netCDF4":
            return self

        self.module = "netCDF4"
        NcFromYaml.__bases__ = (NcCreate,)
        return self

    def use_h5py(self: NcFromYaml) -> NcFromYaml:
        """Use module h5py to generate the HDF5/netCDF4 file."""
        if self.module == "h5py":
            return self

        self.module = "h5py"
        NcFromYaml.__bases__ = (H5Create,)
        return self
