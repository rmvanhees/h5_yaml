#
# This file is part of h5_yaml
#    https://github.com/rmvanhees/h5_yaml.git
#
# Copyright (c) 2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Obtain chunksizes for HDF5 datasets."""

from __future__ import annotations

__all__ = ["guess_chunks"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def guess_chunks(dims: ArrayLike[int], dtype_sz: int) -> str | tuple[int]:
    """Perform an educated guess for the dataset chunk sizes.

    Parameters
    ----------
    dims :  ArrayLike[int]
       Dimensions of the variable
    dtype_sz :  int
       The element size of the data-type of the variable

    Returns
    -------
    "contiguous" or tuple with chunk-sizes

    """
    fixed_size = dtype_sz
    for val in [x for x in dims if x > 0]:
        fixed_size *= val

    if 0 in dims:  # variable with an unlimited dimension
        udim = dims.index(0)
    else:  # variable has no unlimited dimension
        udim = 0
        if fixed_size < 65536:
            return "contiguous"

    if len(dims) == 1:
        return (1024,)

    res = list(dims)
    res[udim] = min(1024, (2048 * 1024) // (fixed_size // max(1, dims[0])))

    return tuple(res)
