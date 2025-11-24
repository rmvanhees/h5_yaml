#
# This file is part of Python package: `h5yaml`
#
#     https://github.com/rmvanhees/pyxarr.git
#
# Copyright (c) 2025 - R.M. van Hees (SRON)
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
"""Read settings from a YAML file."""

from __future__ import annotations

__all__ = ["conf_from_yaml"]

from pathlib import Path

import yaml


# - main function -----------------------------------
def conf_from_yaml(file_path: Path | str) -> dict:
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
        file_path = Path(str)

    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path} not found")

    with file_path.open("r", encoding="ascii") as fid:
        try:
            settings = yaml.safe_load(fid)
        except yaml.YAMLError as exc:
            raise RuntimeError from exc

    return settings
