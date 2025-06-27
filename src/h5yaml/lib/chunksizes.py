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
    if len(dims) > 1:
        for val in [x for x in dims[1:] if x > 0]:
            fixed_size *= val

    # first variables without an unlimited dimension
    if 0 not in dims:
        if fixed_size < 400000:
            return "contiguous"

        res = list(dims)
        res[0] = max(1, 2048000 // fixed_size)
        return tuple(res)

    # then variables with an unlimited dimension
    if len(dims) == 1:
        return (1024,)

    udim = dims.index(0)
    res = list(dims)
    if fixed_size < 400000:
        res[udim] = 1024
    else:
        res[udim] = max(1, 2048000 // fixed_size)

    return tuple(res)
