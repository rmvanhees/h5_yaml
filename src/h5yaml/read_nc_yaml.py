#
# This file is part of Python package: `h5yaml`
#
#     https://github.com/rmvanhees/h5_yaml.git
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
"""Read netCDF4/HDF5 template from YAML configuration file."""

from __future__ import annotations

__all__ = ["ReadNcYaml"]

import pprint
from pathlib import Path

import yaml
import yaml_include


# - local function -------------------------------------
def _from_yaml(file_path: Path | str) -> dict:
    """Read netCDF4/Hdf5 structure from a YAML file.

    Parameters
    ----------
    file_path :  Path | str
       full path of YAML file

    Returns
    -------
    dict
       dictionary with netCDF4 elements (groups, dimensions, variables and attributes)

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
class ReadNcYaml:
    """Class to read and show netCDF4/HDF5 template from YAML configuration file.

    Parameters
    ----------
    nc_yaml_fl :  Path | str | list[Path | str]
       YAML file(s) with the template of a netCDF4/HDF5 file

    """

    def __init__(
        self: ReadNcYaml,
        nc_yaml_fl: Path | str | list[Path | str],
    ) -> None:
        """Construct a ReadNcYaml instance."""
        self.groups = set()
        self.compounds = {}
        self.dimensions = {}
        self.variables = {}
        self.attrs_global = {}
        self.attrs_groups = {}

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

    def __repr__(self: ReadNcYaml) -> str:
        """Show object as dictionary."""
        return pprint.pformat(self.asdict)

    @property
    def asdict(self: ReadNcYaml) -> dict:
        """Return dictionary with netCDF4/HDF5 elements."""
        return {
            "groups": self.groups,
            "dimensions": self.dimensions,
            "compounds": self.compounds,
            "variables": self.variables,
            "attrs_global": self.attrs_global,
            "attrs_groups": self.attrs_groups,
        }
