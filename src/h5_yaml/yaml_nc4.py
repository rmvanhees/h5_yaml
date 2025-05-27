#
# This file is part of h5_yaml
#    https://github.com/rmvanhees/h5_yaml.git"
#
# Copyright (c) 2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Create HDF5/netCDF4 formatted file from a YAML configuration file using netCDF4."""

from __future__ import annotations

__all__ = ["NC4Yaml"]

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4
import numpy as np

from .settings import conf_from_yaml

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# - global parameters ------------------------------


# - local functions --------------------------------


# - class definition -----------------------------------
class NC4Yaml:
    """Class to create a HDF5/netCDF4 formated file from a YAML configuration file."""
