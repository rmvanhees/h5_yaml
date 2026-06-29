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

__all__ = ["Template"]

import pprint
from pathlib import Path

import yaml
import yaml_include


# - local function -------------------------------------
def _from_yaml(file_path: Path | str) -> dict:
    """Read definition of netCDF4/HDF5 elements from a YAML file.

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
class Template:
    """Class to obtain netCDF4/HDF5 elements from YAML file(s) or Python dictionary.

    Parameters
    ----------
    nc_yaml :  list[Path | str] | Path | str | None
       YAML file(s) with netCDF4/HDF5 elements
    nc_dict :  dict[str, dict] | None
       Python dictionary with netCDF4/HDF5 elements

    """

    def __init__(
        self: Template,
        nc_yaml: list[Path | str] | Path | str | None,
        nc_dict: dict[str, dict] | None = None,
    ) -> None:
        """Construct a Template instance."""
        if nc_yaml is not None:
            self.from_yaml(nc_yaml)
        elif nc_dict is not None:
            self.from_dict(nc_dict)
        else:
            self.__unset_attrs__()

    def __repr__(self: Template) -> str:
        """Show object as dictionary."""
        return pprint.pformat(self.asdict)

    def __unset_attrs__(self: Template) -> None:
        """Initialize all class attributes."""
        self.groups = set()
        self.compounds = {}
        self.dimensions = {}
        self.variables = {}
        self.attrs_global = {}
        self.attrs_groups = {}

    def from_dict(self: Template, nc_dict: dict[str, dict]) -> None:
        """Initialize class attributes from a Python dictionary."""
        self.__unset_attrs__()
        if "groups" in nc_dict:
            self.groups = set(nc_dict["groups"])
        if "compounds" in nc_dict:
            self.compounds = nc_dict["compounds"]
        if "dimensions" in nc_dict:
            self.dimensions = nc_dict["dimensions"]
        if "variables" in nc_dict:
            self.variables = nc_dict["variables"]
        if "attrs_global" in nc_dict:
            self.attrs_global = nc_dict["attrs_global"]
        if "attrs_groups" in nc_dict:
            self.attrs_groups = nc_dict["attrs_groups"]

    def from_yaml(self: Template, nc_yaml: list[Path | str] | Path | str) -> None:
        """Initialize class attributes from YAML file(s)."""
        self.__unset_attrs__()
        for yaml_file in nc_yaml if isinstance(nc_yaml, list) else [nc_yaml]:
            try:
                config = _from_yaml(yaml_file)
            except (FileNotFoundError, RuntimeError) as exc:
                raise RuntimeError(f"Fails to access YAML file: {yaml_file}") from exc

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

    @property
    def asdict(self: Template) -> dict:
        """Return dictionary with netCDF4/HDF5 elements."""
        return {
            "groups": self.groups,
            "dimensions": self.dimensions,
            "compounds": self.compounds,
            "variables": self.variables,
            "attrs_global": self.attrs_global,
            "attrs_groups": self.attrs_groups,
        }

    def set_dims(self: Template, dict_dims: dict[str, int]) -> None:
        """Set undefined (= -1) or unlimited dimension (= 0)."""
        for key, value in dict_dims.items():
            if self.dimensions[key]["_size"] <= 0:
                self.dimensions[key]["_size"] = value
