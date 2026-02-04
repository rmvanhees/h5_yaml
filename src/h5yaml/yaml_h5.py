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
"""Create HDF5/netCDF4 formatted file from a YAML configuration file using h5py."""

from __future__ import annotations

__all__ = ["H5Yaml"]

import logging
from pathlib import Path

import h5py
import numpy as np

from .conf_from_yaml import conf_from_yaml
from .lib.adjust_attr import adjust_attr


# - class definition -----------------------------------
class H5Yaml:
    """Class to create a HDF5/netCDF4 formated file from a YAML configuration file.

    Parameters
    ----------
    h5_yaml_fl :  Path | str | list[Path | str]
       YAML files with the HDF5 format definition

    """

    def __init__(self: H5Yaml, h5_yaml_fl: Path | str | list[Path | str]) -> None:
        """Construct a H5Yaml instance."""
        self.logger = logging.getLogger("h5yaml.H5Yaml")
        self._h5_def = {
            "groups": set(),
            "attrs_global": {},
            "attrs_groups": {},
            "compounds": {},
            "dimensions": {},
            "variables": {},
        }

        for yaml_fl in h5_yaml_fl if isinstance(h5_yaml_fl, list) else [h5_yaml_fl]:
            print(yaml_fl)
            try:
                config = conf_from_yaml(yaml_fl)
            except RuntimeError as exc:
                raise RuntimeError from exc
            print(config)
            for key in self._h5_def:
                if key in config:
                    print(key, config[key])
                    self._h5_def[key] |= (
                        set(config[key]) if key == "groups" else config[key]
                    )

    def __attrs(self: H5Yaml, fid: h5py.File) -> None:
        """Create global and group attributes."""
        for key, value in self._h5_def["attrs_global"].items():
            if key not in fid.attrs and value != "TBW":
                fid.attrs[key] = value

        for key, value in self._h5_def["attrs_groups"].items():
            if key not in fid.attrs and value != "TBW":
                fid[str(Path(key).parent)].attrs[Path(key).name] = value

    def __groups(self: H5Yaml, fid: h5py.File) -> None:
        """Create groups in HDF5 product."""
        for key in self._h5_def["groups"]:
            print(f"create group: {key}")
            _ = fid.require_group(key)

    def __dimensions(self: H5Yaml, fid: h5py.File) -> None:
        """Add dimensions to HDF5 product."""
        for key, val in self._h5_def["dimensions"].items():
            fillvalue = None
            if "_FillValue" in val:
                fillvalue = (
                    np.nan if val["_FillValue"] == "NaN" else int(val["_FillValue"])
                )

            if val["_size"] == 0:
                ds_chunk = val.get("_chunks")
                if ds_chunk is not None and not isinstance(ds_chunk, bool):
                    ds_chunk = tuple(ds_chunk)
                dset = fid.create_dataset(
                    key,
                    shape=(0,),
                    dtype="T" if val["_dtype"] == "str" else val["_dtype"],
                    chunks=ds_chunk,
                    maxshape=(None,),
                    fillvalue=fillvalue,
                )
            else:
                dset = fid.create_dataset(
                    key,
                    shape=(val["_size"],),
                    dtype="T" if val["_dtype"] == "str" else val["_dtype"],
                )
                if "_values" in val:
                    dset[:] = val["_values"]
                elif "_range" in val:
                    dset[:] = np.arange(*val["_range"], dtype=val["_dtype"])

            dset.make_scale(
                Path(key).name
                if "long_name" in val
                else "This is a netCDF dimension but not a netCDF variable."
            )
            for attr, attr_val in val.items():
                if not attr.startswith("_"):
                    dset.attrs[attr] = adjust_attr(val["_dtype"], attr, attr_val)

    def __compounds(self: H5Yaml, fid: h5py.File) -> None:
        """Add compound datatypes to HDF5 product."""
        for key, val in self._h5_def["compounds"].items():
            fid[key] = np.dtype([(k, v[0]) for k, v in val.items()])

    def __variables(self: H5Yaml, fid: h5py.File) -> None:
        """Add datasets to HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
           HDF5 file pointer (mode 'r+')

        """
        for key, val in self._h5_def["variables"].items():
            if val["_dtype"] in fid:
                ds_dtype = fid[val["_dtype"]]
            else:
                ds_dtype = "T" if val["_dtype"] == "str" else val["_dtype"]

            fillvalue = None
            if "_FillValue" in val:
                fillvalue = (
                    np.nan if val["_FillValue"] == "NaN" else int(val["_FillValue"])
                )

            # check for scalar dataset
            if val["_dims"][0] == "scalar":
                dset = fid.create_dataset(
                    key,
                    (),
                    dtype=ds_dtype,
                    fillvalue=fillvalue,
                )
                for attr, attr_val in val.items():
                    if not attr.startswith("_"):
                        dset.attrs[attr] = adjust_attr(val["_dtype"], attr, attr_val)
                continue

            n_udim = 0
            ds_shape = ()
            ds_maxshape = ()
            for coord in val["_dims"]:
                dim_sz = fid[coord].size
                n_udim += int(dim_sz == 0)
                ds_shape += (dim_sz,)
                ds_maxshape += (dim_sz if dim_sz > 0 else None,)

            # currently, we can not handle more than one unlimited dimension
            if n_udim > 1:
                raise ValueError(f"{key} has more than one unlimited dimension")

            if None in ds_maxshape and val.get("_chunks") == "contiguous":
                raise KeyError(
                    "you can not create a contiguous dataset with unlimited dimensions."
                )

            # create the variable
            if val.get("_chunks") == "contiguous":
                dset = fid.create_dataset(
                    key,
                    ds_shape,
                    dtype=ds_dtype,
                    chunks=None,
                    maxshape=None,
                    fillvalue=fillvalue,
                )
            else:
                ds_chunk = val.get("_chunks")
                if ds_chunk is not None and not isinstance(ds_chunk, bool):
                    ds_chunk = tuple(ds_chunk)
                compression = None
                shuffle = False
                # currently only gzip compression is supported
                if "_compression" in val:
                    compression = val["_compression"]
                    shuffle = True

                if val.get("_vlen"):
                    ds_name = (
                        val["_dtype"].split("_")[0]
                        if "_" in val["_dtype"]
                        else val["_dtype"]
                    ) + "_vlen"
                    if ds_name not in fid:
                        fid[ds_name] = h5py.vlen_dtype(ds_dtype)
                    ds_dtype = fid[ds_name]
                    fillvalue = None

                dset = fid.create_dataset(
                    key,
                    ds_shape,
                    dtype=ds_dtype,
                    chunks=ds_chunk,
                    maxshape=ds_maxshape,
                    fillvalue=fillvalue,
                    compression=compression,
                    shuffle=shuffle,
                )

            for ii, coord in enumerate(val["_dims"]):
                dset.dims[ii].attach_scale(fid[coord])

            for attr, attr_val in val.items():
                if not attr.startswith("_"):
                    dset.attrs[attr] = adjust_attr(val["_dtype"], attr, attr_val)

            if val["_dtype"] in self._h5_def["compounds"]:
                compound = self._h5_def["compounds"][val["_dtype"]]
                res = [v[2] for k, v in compound.items() if len(v) == 3]
                if res:
                    dset.attrs["units"] = [v[1] for k, v in compound.items()]
                    dset.attrs["long_name"] = res
                else:
                    dset.attrs["long_name"] = [v[1] for k, v in compound.items()]

    @property
    def h5_def(self: H5Yaml) -> dict:
        """Return definition of the HDF5/netCDF4 product."""
        return self._h5_def

    def diskless(self: H5Yaml) -> h5py.File:
        """Create a HDF5/netCDF4 file in memory."""
        fid = h5py.File.in_memory()
        self.__groups(fid)
        self.__dimensions(fid)
        self.__compounds(fid)
        self.__variables(fid)
        self.__attrs(fid)
        return fid

    def create(self: H5Yaml, l1a_name: Path | str) -> None:
        """Create a HDF5/netCDF4 file (overwrite if exist).

        Parameters
        ----------
        l1a_name :  Path | str
           Full name of the HDF5/netCDF4 file to be generated

        """
        try:
            with h5py.File(l1a_name, "w") as fid:
                self.__groups(fid)
                self.__dimensions(fid)
                self.__compounds(fid)
                self.__variables(fid)
                self.__attrs(fid)
        except PermissionError as exc:
            raise RuntimeError(f"failed create {l1a_name}") from exc
