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

from . import __version__
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
        fid.setncattr(
            "_NCCreator",
            (
                f"h5yaml.{self.__class__.__name__}(NcCreate)"
                f",version={__version__.split('+', maxsplit=1)[0]}"
            ),
        )
        for key, val in self.attrs_global.items():
            if val == "TBW":
                continue

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
                    np.nan if value["_FillValue"] == "NaN" else value["_FillValue"]
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

    def __var_scalar(
        self: NcCreate, fid: Dataset, key: str, val: dict, compound: None | dict
    ) -> dict:
        """Return parameters to create a scalar variable.

        Parameters
        ----------
        key :  str
           Name of the variable
        val :  dict
           Properties of the variable
        compound :  None | dict
           Properties of the compound data-type

        """
        fillvalue = None
        if compound is None and "_FillValue" in val:
            fillvalue = np.nan if val["_FillValue"] == "NaN" else int(val["_FillValue"])

        datatype = val["_dtype"]
        if compound is not None:
            cmp_t = np.dtype([(k, *v) for k, v in compound.items()])
            pkey = PurePosixPath(val["_dtype"])
            if pkey.is_absolute():
                datatype = fid[pkey.parent].createCompoundType(cmp_t, pkey.name)
            else:
                datatype = fid.createCompoundType(cmp_t, val["_dtype"])

        return {
            "varname": key,
            "datatype": datatype,
            "fill_value": fillvalue,
        }

    def __var_nochunk(
        self: NcCreate, fid: Dataset, key: str, val: dict, compound: None | dict
    ) -> dict:
        """Return parameters to create a variable without chunking.

        Parameters
        ----------
        key :  str
           Name of the variable
        val :  dict
           Properties of the variable
        compound :  None | dict
           Properties of the compound data-type

        """
        fillvalue = None
        if compound is None and "_FillValue" in val:
            fillvalue = np.nan if val["_FillValue"] == "NaN" else int(val["_FillValue"])

        datatype = val["_dtype"]
        if compound is not None:
            cmp_t = np.dtype([(k, *v) for k, v in compound.items()])
            pkey = PurePosixPath(val["_dtype"])
            if pkey.is_absolute():
                datatype = fid[pkey.parent].createCompoundType(cmp_t, pkey.name)
            else:
                datatype = fid.createCompoundType(cmp_t, val["_dtype"])

        n_udim = 0
        var_dims = []
        for coord in val["_dims"]:
            pcoord = PurePosixPath(coord)
            var_dims.append(pcoord.name if pcoord.is_absolute() else coord)
            if pcoord.is_absolute():
                dim_sz = fid[pcoord.parent].dimensions[pcoord.name].size
            else:
                dim_sz = fid.dimensions[coord].size
            n_udim += int(dim_sz == 0)

        # currently, we can not handle more than one unlimited dimension
        if n_udim > 0:
            raise KeyError(
                "you can not create a contiguous dataset with unlimited dimensions."
            )

        return {
            "varname": key,
            "datatype": datatype,
            "dimensions": var_dims,
            "contiguous": True,
            "fill_value": fillvalue,
        }

    def __var_chunked(
        self: NcCreate, fid: Dataset, key: str, val: dict, compound: None | dict
    ) -> dict:
        """Return parameters to create a variable with chunking.

        Parameters
        ----------
        key :  str
           Name of the variable
        val :  dict
           Properties of the variable
        compound :  None | dict
           Properties of the compound data-type

        """
        fillvalue = None
        if compound is None and "_FillValue" in val:
            fillvalue = np.nan if val["_FillValue"] == "NaN" else int(val["_FillValue"])

        n_udim = 0
        var_dims = []
        for coord in val["_dims"]:
            pcoord = PurePosixPath(coord)
            var_dims.append(pcoord.name if pcoord.is_absolute() else coord)
            if pcoord.is_absolute():
                dim_sz = fid[pcoord.parent].dimensions[pcoord.name].size
            else:
                dim_sz = fid.dimensions[coord].size
            n_udim += int(dim_sz == 0)

        # currently, we can not handle more than one unlimited dimension
        if n_udim > 1:
            raise ValueError(f"{key} has more than one unlimited dimension")

        compression = None
        complevel = 0
        # currently only gzip compression is supported
        if "_compression" in val:
            compression = "zlib"
            complevel = val["_compression"]

        ds_chunk = val.get("_chunks")
        if ds_chunk is not None:
            ds_chunk = None if isinstance(ds_chunk, bool) else tuple(ds_chunk)

        datatype = val["_dtype"]
        if compound is not None:
            cmp_t = np.dtype([(k, *v) for k, v in compound.items()])
            pkey = PurePosixPath(val["_dtype"])
            if pkey.is_absolute():
                datatype = fid[pkey.parent].createCompoundType(cmp_t, pkey.name)
            else:
                datatype = fid.createCompoundType(cmp_t, val["_dtype"])

        if "_vlen" in val:
            if val["_dtype"] in fid.cmptypes:
                raise ValueError("can not have vlen with compounds")
            datatype = fid.createVLType(datatype, "phony_vlen")
            fillvalue = None

        return {
            "varname": key,
            "datatype": datatype,
            "dimensions": var_dims,
            "fill_value": fillvalue,
            "compression": compression,
            "complevel": complevel,
            "chunksizes": ds_chunk,
        }

    def __variables(self: NcCreate, fid: Dataset) -> None:
        """Add datasets to a netCDF4 product.

        Parameters
        ----------
        fid :  netCDF4.Dataset
           netCDF4 Dataset (mode 'r+')

        """
        for key, val in self.variables.items():
            pkey = PurePosixPath(key)
            var_grp = fid[pkey.parent] if pkey.is_absolute() else fid
            var_name = pkey.name if pkey.is_absolute() else key

            # check if dtype of variable is a compound
            compound = self.compounds.get(val["_dtype"], None)

            # create variable
            if val["_dims"][0] == "scalar":
                dset = var_grp.createVariable(
                    **self.__var_scalar(fid, var_name, val, compound)
                )
                print(f"scalar: {dset}")
            elif val.get("_chunks") == "contiguous":
                dset = var_grp.createVariable(
                    **self.__var_nochunk(fid, var_name, val, compound)
                )
            else:
                dset = var_grp.createVariable(
                    **self.__var_chunked(fid, var_name, val, compound)
                )

            # write data to dataset
            if "_values" in val:
                dset[:] = val["_values"]

            # add user-supplied attributes
            dset.setncatts(
                {
                    k: adjust_attr(val["_dtype"], k, v)
                    for k, v in val.items()
                    if not k.startswith("_")
                }
            )

    def create(self: NcCreate, filename: Path | str, mode: str = "w") -> None:
        """Create a netCDF4 file (overwrite if exist).

        Parameters
        ----------
        filename :  Path | str
           Name of the file on disk, or file-like object
        mode :  {"r+", "w", "x", "a"}, default="w"
           The value of mode is passed to netCDF4.Dataset, see `netCDF4` documentation

        """
        try:
            with Dataset(filename, mode) as fid:
                self.__groups(fid)
                self.__dimensions(fid)
                self.__variables(fid)
                self.__attrs(fid)
        except PermissionError as exc:
            raise RuntimeError(f"failed to create {filename}") from exc

    def diskless(self: NcCreate) -> Dataset:
        """Create a netCDF4 file in memory.

        Notes
        -----
        An in-memory file will always created from scratch, `mode` has no meaning.

        Returns
        -------
          Dataset: to add data to the empty netCDF4 file.

        """
        fid = Dataset("diskless.nc", "w", memory=4096)
        self.__groups(fid)
        self.__dimensions(fid)
        self.__variables(fid)
        self.__attrs(fid)
        return fid

    def to_disk(self: NcCreate, fid: Dataset, filename: Path | str) -> None:
        """Write in-memory buffer to file, and close the Dataset.

        Parameters
        ----------
        fid :  Dataset
           Dataset pointer returned by method `diskless`
        filename :  Path | str
           Name of the file on disk, or file-like object

        """
        try:
            with open(filename, "wb") as ff:
                _ = ff.write(fid.close())
        except PermissionError as exc:
            raise RuntimeError(f"failed create {filename}") from exc
        except OSError as exc:
            raise RuntimeError(f"failed to write {filename}") from exc
