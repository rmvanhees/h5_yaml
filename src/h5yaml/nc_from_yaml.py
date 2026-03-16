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

import logging
from pathlib import Path

import yaml

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

    with file_path.open("r", encoding="ascii") as fid:
        try:
            settings = yaml.safe_load(fid)
        except yaml.parser.ParserError as exc:
            raise RuntimeError(f"Failed to parse {file_path}") from exc

    return settings


# - class definition -----------------------------------
class NcFromYaml:
    """Class to create a netCDF4 formated file from a YAML configuration file.

    Parameters
    ----------
    nc_yaml_fl :  Path | str | list[Path | str]
       YAML files with the HDF5 format definition

    """

    def __init__(self: NcFromYaml, nc_yaml_fl: Path | str | list[Path | str]) -> None:
        """Construct a NcFromYaml instance."""
        self.logger = logging.getLogger("h5yaml.NcFromYaml")
        self._nc_def = {
            "groups": None,
            "attrs_global": None,
            "attrs_groups": None,
            "compounds": None,
            "dimensions": None,
            "variables": None,
        }

        for yaml_fl in nc_yaml_fl if isinstance(nc_yaml_fl, list) else [nc_yaml_fl]:
            try:
                config = _from_yaml(yaml_fl)
            except RuntimeError as exc:
                raise RuntimeError from exc

            for key, value in self._nc_def.items():
                if key not in config:
                    continue

                if value is None:
                    value = set() if key == "groups" else {}
                self._nc_def[key] = (
                    value | set(config[key]) if key == "groups" else config[key]
                )

    @property
    def asdict(self: NcFromYaml) -> dict:
        """Return definition of the HDF5/netCDF4 product."""
        return self._nc_def

    def diskless(self: NcFromYaml, module: str = "h5py", persist: bool = False) -> None:
        """Create a HDF5/netCDF4 file in memory."""
        fid = H5Create(**self._nc_def) if module == "h5py" else NcCreate(**self._nc_def)

        return fid.diskless(persist)

    def create(self: NcFromYaml, name: str | Path, module: str = "h5py") -> None:
        """Create a netCDF4 file (overwrite if exist).

        Parameters
        ----------
        name :  Path | str
           Full name of the netCDF4 file to be generated
        module :  {"h5py", "netCDF4"}, default="h5py"
           Use this module to generate the netCDF4 file

        """
        fid = H5Create(**self._nc_def) if module == "h5py" else NcCreate(**self._nc_def)

        fid.create(name)
