#
# This file is part of h5_yaml:
#    https://github.com/rmvanhees/h5_yaml.git
#
# Copyright (c) 2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Read settings from a YAML file."""

from __future__ import annotations

__all__ = ["conf_from_yaml"]

from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path


# - main function -----------------------------------
def conf_from_yaml(file_path: Path) -> dict:
    """Read settings from a YAML file: `file_path`.

    Parameters
    ----------
    file_path :  Path
       full path of YAML file

    Returns
    -------
    dict
       content of the configuration file

    """
    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path} not found")

    with file_path.open("r", encoding="ascii") as fid:
        try:
            settings = yaml.safe_load(fid)
        except yaml.YAMLError as exc:
            raise RuntimeError from exc

    return settings
