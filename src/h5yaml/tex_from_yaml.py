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
"""Convert YAML configuration of a netCDF4 file as LaTeX table."""

from __future__ import annotations

__all__ = ["TexFromYaml"]

from importlib.resources import files
from pathlib import Path, PurePosixPath

import yaml
import yaml_include


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


def _str_dtype(short_name: str) -> str:
    """Convert dtype abriviation to long name."""
    res = short_name
    match short_name:
        case "i1":
            res = "int8"
        case "i2":
            res = "int16"
        case "i4":
            res = "int32"
        case "i8":
            res = "int64"
        case "u1":
            res = "uint8"
        case "u2":
            res = "uint16"
        case "u4":
            res = "uint32"
        case "u8":
            res = "uint64"
        case "f2":
            res = "float16"
        case "f4":
            res = "float32"
        case "f8":
            res = "float64"

    return res


# - class definition -----------------------------------
class TexFromYaml:
    """Class to show the layout of a netCDF4 file in a LaTeX table.

    Parameters
    ----------
    nc_yaml_fl :  Path | str | list[Path | str]
       YAML files with the HDF5 format definition

    """

    def __init__(self: TexFromYaml, nc_yaml_fl: Path | str | list[Path | str]) -> None:
        """Construct a TexFromYaml instance."""
        self.groups = set()
        self.compounds = {}
        self.dimensions = {}
        self.variables = {}

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

    def to_table(
        self: TexFromYaml, caption: str | None = None, label: str | None = None
    ) -> str:
        """..."""
        config = _from_yaml(files("h5yaml.Data") / "tex_table.yaml")
        for line in config["header"]:
            if "MYCAPTION" in line and caption is not None:
                line = line.replace("MYCAPTION", caption)
            if "MYLABEL" in line and label is not None:
                line = line.replace("MYLABEL", label)
            print(line)

        for key, val in self.dimensions.items():
            if key.startswith("/"):
                continue
            print(
                f"{key} & DIM & {_str_dtype(val['_dtype'])} & {val['_size']} & & \\\\"
            )
        for key, val in self.variables.items():
            if key.startswith("/"):
                continue
            print(
                f"{key} & VAR & {_str_dtype(val['_dtype'])} & {val['_size']} & &"
                f" {val.get('long_name', '')} & \\\\"
            )
        for name in sorted(self.groups):
            grp = PurePosixPath("/") / name
            print(f"{grp} & GRP & & & & \\\\")
            for key, val in self.dimensions.items():
                pkey = PurePosixPath(key)
                if pkey.parent == grp:
                    print(
                        f"\\qquad {pkey.name} & DIM & {_str_dtype(val['_dtype'])}"
                        f" & {val['_size']} & & \\\\"
                    )
            for key, val in self.variables.items():
                pkey = PurePosixPath(key)
                if pkey.parent == grp:
                    dims = 1 if val["_dims"][0] == "scalar" else val["_dims"]
                    print(
                        f"\\qquad {pkey.name} & VAR & {_str_dtype(val['_dtype'])}"
                        f" & {dims} & {val.get('units', '1')}"
                        f" & {val.get('long_name', '')} \\\\"
                    )

        for line in config["footer"]:
            print(line)


def main() -> None:
    """..."""
    tex = TexFromYaml(files("h5yaml.Data") / "h5_testing.yaml")
    tex.to_table(caption="TANGO Carbon Level-1A", label="tab:l1a")


if __name__ == "__main__":
    main()
