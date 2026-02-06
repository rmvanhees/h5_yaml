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
"""Convert values HDF5/netCDF4 attribute."""

from __future__ import annotations

__all__ = ["adjust_attr"]

import numpy as np


# - main function ------------------------------------
def adjust_attr(dtype: str, attr_key: str, attr_val: np.generic) -> np.generic:
    """Return attribute converted to the same data type as its variable.

    Parameters
    ----------
    dtype :  str
      numpy data-type of variable
    attr_key :  str
      name of the attribute
    attr_val :  np.generic
      original value of the attribute

    Returns
    -------
    attr_val converted to dtype
    """
    if attr_key == "flag_values":
        return np.array(attr_val, dtype=dtype)

    if attr_key == "flag_masks":
        return np.array(attr_val, dtype=dtype)

    if attr_key in ("valid_min", "valid_max", "valid_range"):
        match dtype:
            case "i1":
                res = np.int8(attr_val)
            case "i2":
                res = np.int16(attr_val)
            case "i4":
                res = np.int32(attr_val)
            case "i8":
                res = np.int64(attr_val)
            case "u1":
                res = np.uint8(attr_val)
            case "u2":
                res = np.uint16(attr_val)
            case "u4":
                res = np.uint32(attr_val)
            case "u8":
                res = np.uint64(attr_val)
            case "f2":
                res = np.float16(attr_val)
            case "f4":
                res = np.float32(attr_val)
            case "f8":
                res = np.float64(attr_val)
            case _:
                res = attr_val

        return res

    return attr_val
