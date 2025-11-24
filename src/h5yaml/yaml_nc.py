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
"""Create HDF5/netCDF4 formatted file from a YAML configuration file using netCDF4."""

from __future__ import annotations

__all__ = ["NcYaml"]

import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import numpy as np

# pylint: disable=no-name-in-module
from netCDF4 import Dataset

from .conf_from_yaml import conf_from_yaml
from .lib.adjust_attr import adjust_attr
from .lib.chunksizes import guess_chunks

if TYPE_CHECKING:
    from pathlib import Path


# - class definition -----------------------------------
class NcYaml:
    """Class to create a HDF5/netCDF4 formated file from a YAML configuration file."""

    def __init__(self: NcYaml, nc_yaml_fl: Path) -> None:
        """Construct a NcYaml instance."""
        self.logger = logging.getLogger("h5yaml.NcYaml")

        try:
            self._nc_def = conf_from_yaml(nc_yaml_fl)
        except RuntimeError as exc:
            raise RuntimeError from exc

        self.yaml_dir = nc_yaml_fl.parent

    def __groups(self: NcYaml, fid: Dataset) -> None:
        """Create groups in HDF5 product."""
        for key in self.nc_def["groups"]:
            pkey = PurePosixPath(key)
            if pkey.is_absolute():
                _ = fid[pkey.parent].createGroup(pkey.name)
            else:
                _ = fid.createGroup(key)

    def __dimensions(self: NcYaml, fid: Dataset) -> None:
        """Add dimensions to HDF5 product."""
        for key, value in self.nc_def["dimensions"].items():
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

    def __compounds(self: NcYaml, fid: Dataset) -> dict[str, str | int | float]:
        """Add compound datatypes to HDF5 product."""
        if "compounds" not in self.nc_def:
            return {}

        compounds = {}
        if isinstance(self.nc_def["compounds"], list):
            file_list = self.nc_def["compounds"].copy()
            self.nc_def["compounds"] = {}
            for name in file_list:
                if not (yaml_fl := self.yaml_dir / name).is_file():
                    continue
                try:
                    res = conf_from_yaml(yaml_fl)
                except RuntimeError as exc:
                    raise RuntimeError from exc
                for key, value in res.items():
                    self.nc_def["compounds"][key] = value

        for key, value in self.nc_def["compounds"].items():
            compounds[key] = {
                "dtype": [],
                "units": [],
                "names": [],
            }

            for _key, _val in value.items():
                compounds[key]["dtype"].append((_key, _val[0]))
                if len(_val) == 3:
                    compounds[key]["units"].append(_val[1])
                compounds[key]["names"].append(_val[2] if len(_val) == 3 else _val[1])

            comp_t = np.dtype(compounds[key]["dtype"])
            _ = fid.createCompoundType(comp_t, key)

        return compounds

    def __variables(
        self: NcYaml,
        fid: Dataset,
        compounds: dict[str, str | int | float] | None,
    ) -> None:
        """Add datasets to HDF5 product.

        Parameters
        ----------
        fid :  netCDF4.Dataset
           HDF5 file pointer (mode 'r+')
        compounds :  dict[str, str | int | float]
           Definition of the compound(s) in the product

        """
        for key, val in self.nc_def["variables"].items():
            pkey = PurePosixPath(key)
            var_grp = fid[pkey.parent] if pkey.is_absolute() else fid
            var_name = pkey.name if pkey.is_absolute() else key

            if val["_dtype"] in fid.cmptypes:
                ds_dtype = fid.cmptypes[val["_dtype"]].dtype
                sz_dtype = ds_dtype.itemsize
            else:
                ds_dtype = val["_dtype"]
                sz_dtype = np.dtype(val["_dtype"]).itemsize

            fillvalue = None
            if "_FillValue" in val:
                fillvalue = (
                    np.nan if val["_FillValue"] == "NaN" else int(val["_FillValue"])
                )

            # check for scalar dataset
            if val["_dims"][0] == "scalar":
                dset = var_grp.createVariable(
                    var_name,
                    val["_dtype"],
                    fill_value=fillvalue,
                    contiguous=True,
                )
                dset.setncatts(
                    {
                        k: adjust_attr(val["_dtype"], k, v)
                        for k, v in val.items()
                        if not k.startswith("_")
                    }
                )
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

            # obtain chunk-size settings
            ds_chunk = (
                val["_chunks"] if "_chunks" in val else guess_chunks(ds_shape, sz_dtype)
            )

            if val["_dtype"] in fid.cmptypes:
                val["_dtype"] = fid.cmptypes[val["_dtype"]]

            # create the variable
            if ds_chunk == "contiguous":
                dset = var_grp.createVariable(
                    var_name,
                    val["_dtype"],
                    dimensions=var_dims,
                    fill_value=fillvalue,
                    contiguous=True,
                )
            else:
                if val.get("_vlen"):
                    if val["_dtype"] in fid.cmptypes:
                        raise ValueError("can not have vlen with compounds")
                    val["_dtype"] = fid.createVLType(ds_dtype, val["_dtype"])
                    fillvalue = None
                    if ds_maxshape == (None,):
                        ds_chunk = (16,)

                dset = var_grp.createVariable(
                    var_name,
                    str if val["_dtype"] == "str" else val["_dtype"],
                    dimensions=var_dims,
                    fill_value=fillvalue,
                    contiguous=False,
                    compression=compression,
                    complevel=complevel,
                    chunksizes=(
                        ds_chunk if isinstance(ds_chunk, tuple) else tuple(ds_chunk)
                    ),
                )
            dset.setncatts(
                {
                    k: adjust_attr(val["_dtype"], k, v)
                    for k, v in val.items()
                    if not k.startswith("_")
                }
            )

            if compounds is not None and val["_dtype"] in compounds:
                if compounds[val["_dtype"]]["units"]:
                    dset.attrs["units"] = compounds[val["_dtype"]]["units"]
                if compounds[val["_dtype"]]["names"]:
                    dset.attrs["long_name"] = compounds[val["_dtype"]]["names"]

    @property
    def nc_def(self: NcYaml) -> dict:
        """Return definition of the HDF5/netCDF4 product."""
        return self._nc_def

    def diskless(self: NcYaml) -> Dataset:
        """Create a HDF5/netCDF4 file in memory."""
        fid = Dataset("diskless_test.nc", "w", diskless=True, persistent=False)
        self.__groups(fid)
        self.__dimensions(fid)
        self.__variables(fid, self.__compounds(fid))
        return fid

    def create(self: NcYaml, l1a_name: Path | str) -> None:
        """Create a HDF5/netCDF4 file (overwrite if exist).

        Parameters
        ----------
        l1a_name :  Path | str
           Full name of the HDF5/netCDF4 file to be generated

        """
        try:
            with Dataset(l1a_name, "w") as fid:
                self.__groups(fid)
                self.__dimensions(fid)
                self.__variables(fid, self.__compounds(fid))
        except PermissionError as exc:
            raise RuntimeError(f"failed to create {l1a_name}") from exc
