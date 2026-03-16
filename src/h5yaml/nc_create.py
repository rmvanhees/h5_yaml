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
"""Initialize empty netCDF4 file using Python package `netCDF4`."""

from __future__ import annotations

__all__ = ["NcCreate"]

import logging
from pathlib import Path, PurePosixPath

import numpy as np

# pylint: disable=no-name-in-module
from netCDF4 import Dataset

from .lib.adjust_attr import adjust_attr


# - class definition -----------------------------------
class NcCreate:
    """Class to create an empty netCDF4 file using Python package `netCDF4`.

    Parameters
    ----------
    groups: set | None = None
    compounds: dict | None = None
    dimensions: dict | None = None
    variables: dict | None = None
    attrs_global: dict | None = None
    attrs_groups: dict | None = None

    """

    def __init__(
        self: NcCreate,
        groups: set | None = None,
        compounds: dict | None = None,
        dimensions: dict | None = None,
        variables: dict | None = None,
        attrs_global: dict | None = None,
        attrs_groups: dict | None = None,
    ) -> None:
        """Construct a NcCreate instance."""
        self.logger = logging.getLogger("h5yaml.NcCreate")
        self.groups = set() if groups is None else groups
        self.compounds = {} if compounds is None else compounds
        self.dimensions = {} if dimensions is None else dimensions
        self.variables = {} if variables is None else variables
        self.attrs_global = {} if attrs_global is None else attrs_global
        self.attrs_groups = {} if attrs_groups is None else attrs_groups

    def __attrs(self: NcCreate, fid: Dataset) -> None:
        """Create global and group attributes.

        Parameters
        ----------
        fid :  netCDF4.Dataset
           netCDF4 Dataset (mode 'r+')

        """
        for key, val in self.attrs_global.items():
            if val == "TBW":
                continue

            if isinstance(val, str):
                fid.setncattr_string(key, val)
            else:
                fid.setncattr(key, val)

    def __groups(self: NcCreate, fid: Dataset) -> None:
        """Create groups in a netCDF4 product.

        Parameters
        ----------
        fid :  netCDF4.Dataset
           netCDF4 Dataset (mode 'r+')

        """
        for key in self.groups:
            _ = fid.createGroup(key)

        for key, val in self.attrs_groups.items():
            if val == "TBW":
                continue

            if isinstance(val, str):
                fid[str(Path(key).parent)].setncattr_string(Path(key).name, val)
            else:
                fid[str(Path(key).parent)].setncattr(Path(key).name, val)

    def __dimensions(self: NcCreate, fid: Dataset) -> None:
        """Add dimensions to a netCDF4 product.

        Parameters
        ----------
        fid :  netCDF4.Dataset
           netCDF4 Dataset (mode 'r+')

        """
        for key, value in self.dimensions.items():
            pkey = PurePosixPath(key)
            if pkey.is_absolute():
                _ = fid[pkey.parent].createDimension(pkey.name, value["_size"])
            else:
                _ = fid.createDimension(key, value["_size"])

            if len(value) <= 2:
                continue

            fillvalue = None
            if "_FillValue" in value:
                fillvalue = (
                    np.nan if value["_FillValue"] == "NaN" else int(value["_FillValue"])
                )

            if pkey.is_absolute():
                dset = fid[pkey.parent].createVariable(
                    pkey.name,
                    value["_dtype"],
                    dimensions=(pkey.name,),
                    fill_value=fillvalue,
                    contiguous=value["_size"] != 0,
                )
            else:
                dset = fid.createVariable(
                    key,
                    value["_dtype"],
                    dimensions=(key,),
                    fill_value=fillvalue,
                    contiguous=value["_size"] != 0,
                )
            if value["_size"] > 0:
                if "_values" in value:
                    dset[:] = np.array(value["_values"])
                elif "_range" in value:
                    dset[:] = np.arange(*value["_range"], dtype=value["_dtype"])

            dset.setncatts(
                {
                    k: adjust_attr(value["_dtype"], k, v)
                    for k, v in value.items()
                    if not k.startswith("_")
                }
            )

    def __compounds(self: NcCreate, fid: Dataset) -> None:
        """Add compound datatypes to a netCDF4 product.

        Parameters
        ----------
        fid :  netCDF4.Dataset
           netCDF4 Dataset (mode 'r+')

        """
        for key, val in self.compounds.items():
            comp_t = np.dtype([(k, v[0]) for k, v in val.items()])
            _ = fid.createCompoundType(comp_t, key)

    def __variables(self: NcCreate, fid: Dataset) -> None:
        """Add datasets to a netCDF4 product.

        Parameters
        ----------
        fid :  netCDF4.Dataset
           netCDF4 Dataset (mode 'r+')

        """

        def add_compound_attr() -> None:
            compound = self.compounds[val["_dtype"]]
            res = [v[2] for k, v in compound.items() if len(v) == 3]
            if res:
                dset.units = [v[1] for k, v in compound.items()]
                dset.names = res
            else:
                dset.names = [v[1] for k, v in compound.items()]

        for key, val in self.variables.items():
            pkey = PurePosixPath(key)
            var_grp = fid[pkey.parent] if pkey.is_absolute() else fid
            var_name = pkey.name if pkey.is_absolute() else key

            if val["_dtype"] in fid.cmptypes:
                is_compound = True
                ds_dtype = fid.cmptypes[val["_dtype"]]
            else:
                is_compound = False
                ds_dtype = val["_dtype"]

            fillvalue = None
            if "_FillValue" in val:
                fillvalue = (
                    np.nan if val["_FillValue"] == "NaN" else int(val["_FillValue"])
                )

            # check for scalar dataset
            if val["_dims"][0] == "scalar":
                dset = var_grp.createVariable(
                    var_name,
                    ds_dtype,
                    fill_value=fillvalue,
                    contiguous=True,
                )
                dset.setncatts(
                    {
                        k: adjust_attr(ds_dtype, k, v)
                        for k, v in val.items()
                        if not k.startswith("_")
                    }
                )
                if is_compound:
                    add_compound_attr()
                continue

            compression = None
            complevel = 0
            # currently only gzip compression is supported
            if "_compression" in val:
                compression = "zlib"
                complevel = val["_compression"]

            var_dims = []
            n_udim = 0
            ds_shape = ()
            ds_maxshape = ()
            for coord in val["_dims"]:
                pcoord = PurePosixPath(coord)
                var_dims.append(pcoord.name if pcoord.is_absolute() else coord)
                if pcoord.is_absolute():
                    dim_sz = fid[pcoord.parent].dimensions[pcoord.name].size
                else:
                    dim_sz = fid.dimensions[coord].size
                n_udim += int(dim_sz == 0)
                ds_shape += (dim_sz,)
                ds_maxshape += (dim_sz if dim_sz > 0 else None,)

            # currently, we can not handle more than one unlimited dimension
            if n_udim > 1:
                raise ValueError("more than one unlimited dimension")

            if None in ds_maxshape and val.get("_chunks") == "contiguous":
                raise KeyError(
                    "you can not create a contiguous dataset with unlimited dimensions."
                )

            # create the variable
            if val.get("_chunks") == "contiguous":
                dset = var_grp.createVariable(
                    var_name,
                    ds_dtype,
                    dimensions=var_dims,
                    fill_value=fillvalue,
                    contiguous=True,
                )
            else:
                ds_chunk = val.get("_chunks")
                if ds_chunk is not None:
                    ds_chunk = None if isinstance(ds_chunk, bool) else tuple(ds_chunk)
                if val.get("_vlen"):
                    if is_compound:
                        raise ValueError("can not have vlen with compounds")
                    ds_dtype = fid.createVLType(ds_dtype, "phony_vlen")
                    fillvalue = None

                dset = var_grp.createVariable(
                    var_name,
                    str if val["_dtype"] == "str" else ds_dtype,
                    dimensions=var_dims,
                    fill_value=fillvalue,
                    compression=compression,
                    complevel=complevel,
                    chunksizes=ds_chunk,
                    contiguous=False,
                )
            dset.setncatts(
                {
                    k: adjust_attr(val["_dtype"], k, v)
                    for k, v in val.items()
                    if not k.startswith("_")
                }
            )
            if is_compound:
                add_compound_attr()

    def diskless(self: NcCreate, persist: bool) -> Dataset:
        """Create a netCDF4 file in memory."""
        fid = Dataset("diskless_test.nc", "w", diskless=True, persist=persist)
        self.__groups(fid)
        self.__dimensions(fid)
        self.__compounds(fid)
        self.__variables(fid)
        self.__attrs(fid)
        return fid

    def create(self: NcCreate, l1a_name: Path | str) -> None:
        """Create a netCDF4 file (overwrite if exist).

        Parameters
        ----------
        l1a_name :  Path | str
           Full name of the netCDF4 file to be generated

        """
        try:
            with Dataset(l1a_name, "w") as fid:
                self.__groups(fid)
                self.__dimensions(fid)
                self.__compounds(fid)
                self.__variables(fid)
                self.__attrs(fid)
        except PermissionError as exc:
            raise RuntimeError(f"failed to create {l1a_name}") from exc
