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
"""Initialize empty netCDF4 file using Python package `h5py`."""

from __future__ import annotations

__all__ = ["H5Create"]

import logging
from pathlib import Path

import h5py
import numpy as np

from . import sw_version
from .lib.safe_eval import safe_eval

H5_LIBVER = ("v110", "latest")


def str2bytes(ss: str) -> np.bytes_:
    """Convert Python string to ASCII encoded bytes."""
    return np.array(ss, dtype=h5py.string_dtype("ascii", len(ss)))


# - class definition -----------------------------------
class H5Create:
    """Class to create an empty netCDF4 file using Python package `h5py`.

    Parameters
    ----------
    groups: set | None = None
    compounds: dict | None = None
    dimensions: dict | None = None
    variables: dict | None = None
    attrs_global: dict | None = None
    attrs_groups: dict | None = None

    """

    str2bytes = False

    def __init__(
        self: H5Create,
        groups: set | None = None,
        compounds: dict | None = None,
        dimensions: dict | None = None,
        variables: dict | None = None,
        attrs_global: dict | None = None,
        attrs_groups: dict | None = None,
    ) -> None:
        """Construct a H5Create instance."""
        self.logger = logging.getLogger("h5yaml.H5Create")
        self.groups = set() if groups is None else groups
        self.compounds = {} if compounds is None else compounds
        self.dimensions = {} if dimensions is None else dimensions
        self.variables = {} if variables is None else variables
        self.attrs_global = {} if attrs_global is None else attrs_global
        self.attrs_groups = {} if attrs_groups is None else attrs_groups

    def _adjust_attr(
        self: H5Create, dtype: str, attr_key: str, attr_val: np.generic
    ) -> np.generic:
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

        if attr_key == "scale_factor" and isinstance(attr_val, str):
            return safe_eval(attr_val)

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

        if self.str2bytes and isinstance(attr_val, str):
            return str2bytes(attr_val)

        return attr_val

    def __attrs(self: H5Create, fid: h5py.File) -> None:
        """Create global and group attributes.

        Parameters
        ----------
        fid :  h5py.File
           HDF5 file pointer (mode 'w' or 'r+')

        """
        value = (
            f"h5yaml.{self.__class__.__name__}(H5Create),version={sw_version()}"
            f",{'options=str_as_bytes' if self.str2bytes else ''}"
        )
        fid.attrs["_NCCreator"] = str2bytes(value) if self.str2bytes else value
        value = f"version=2,hdf5={h5py.version.hdf5_version}"
        fid.attrs["_NCProperties"] = str2bytes(value) if self.str2bytes else value
        for key, value in self.attrs_global.items():
            if key in fid.attrs or value == "TBW":
                continue
            if isinstance(value, str):
                fid.attrs[key] = str2bytes(value) if self.str2bytes else value
            else:
                fid.attrs[key] = value

    def __groups(self: H5Create, fid: h5py.File) -> None:
        """Create groups in HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
           HDF5 file pointer (mode 'w' or 'r+')

        """
        for key in self.groups:
            if key not in fid:
                _ = fid.create_group(key, track_order=True)
        for key, value in self.attrs_groups.items():
            if key in fid.attrs or value == "TBW":
                continue
            if isinstance(value, str):
                fid[str(Path(key).parent)].attrs[Path(key).name] = (
                    str2bytes(value) if self.str2bytes else value
                )
            else:
                fid[str(Path(key).parent)].attrs[Path(key).name] = value

    def __dimensions(self: H5Create, fid: h5py.File) -> None:
        """Add dimensions to HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
         HDF5 file pointer (mode 'w' or 'r+')

        """
        for key, val in self.dimensions.items():
            fillvalue = None
            if "_FillValue" in val:
                fillvalue = np.nan if val["_FillValue"] == "NaN" else val["_FillValue"]

            if val["_size"] == 0:
                dset = fid.create_dataset(
                    key,
                    shape=(0,),
                    dtype="T" if val["_dtype"] == "str" else val["_dtype"],
                    chunks=True,
                    maxshape=(None,),
                    fillvalue=fillvalue,
                )
            else:
                dset = fid.create_dataset(
                    key,
                    shape=(val["_size"],),
                    dtype="T" if val["_dtype"] == "str" else val["_dtype"],
                    fillvalue=fillvalue,
                )
                if "_values" in val:
                    dset[:] = val["_values"]
                elif "_range" in val:
                    dset[:] = np.arange(*val["_range"], dtype=val["_dtype"])

            if fillvalue is not None:
                dset.attrs["_FillValue"] = dset.fillvalue

            dset.make_scale(
                Path(key).name
                if len(val) > 2
                else "This is a netCDF dimension but not a netCDF variable."
            )

            for attr, attr_val in val.items():
                if not attr.startswith("_"):
                    dset.attrs[attr] = self._adjust_attr(val["_dtype"], attr, attr_val)

    def __compounds(self: H5Create, fid: h5py.File) -> None:
        """Add compound datatypes to HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
           HDF5 file pointer (mode 'w' or 'r+')

        """
        for key, val in self.compounds.items():
            fid[key] = np.dtype([(k, *v) for k, v in val.items()], align=True)

    def __var_scalar(self: H5Create, fid: h5py.File, key: str, val: dict) -> dict:
        """Return parameters to create a scalar variable.

        Parameters
        ----------
        key :  str
           Name of the variable
        val :  dict
           Properties of the variable

        """
        fillvalue = None
        if "_FillValue" in val:
            fillvalue = np.nan if val["_FillValue"] == "NaN" else val["_FillValue"]

        return {
            "name": key,
            "shape": (),
            "dtype": fid[val["_dtype"]].dtype
            if val["_dtype"] in fid
            else ("T" if val["_dtype"] == "str" else val["_dtype"]),
            "fillvalue": fillvalue,
        }

    def __var_nochunk(self: H5Create, fid: h5py.File, key: str, val: dict) -> dict:
        """Return parameters to create a variable without chunking.

        Parameters
        ----------
        key :  str
           Name of the variable
        val :  dict
           Properties of the variable

        """
        fillvalue = None
        if "_FillValue" in val:
            fillvalue = np.nan if val["_FillValue"] == "NaN" else val["_FillValue"]

        n_udim = 0
        ds_shape = ()
        for coord in val["_dims"]:
            dim_sz = fid[coord].size
            n_udim += int(dim_sz == 0)
            ds_shape += (dim_sz,)

        if n_udim > 0:
            raise KeyError(
                "you can not create a contiguous dataset with unlimited dimensions."
            )

        return {
            "name": key,
            "shape": ds_shape,
            "maxshape": None,
            "dtype": fid[val["_dtype"]].dtype
            if val["_dtype"] in fid
            else ("T" if val["_dtype"] == "str" else val["_dtype"]),
            "fillvalue": fillvalue,
        }

    def __var_chunked(self: H5Create, fid: h5py.File, key: str, val: dict) -> dict:
        """Return parameters to create a variable with chunking.

        Parameters
        ----------
        key :  str
           Name of the variable
        val :  dict
           Properties of the variable

        """
        fillvalue = None
        if "_FillValue" in val:
            fillvalue = np.nan if val["_FillValue"] == "NaN" else val["_FillValue"]

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

        ds_chunk = val.get("_chunks")
        if ds_chunk is not None and not isinstance(ds_chunk, bool):
            ds_chunk = tuple(ds_chunk)
        compression = None
        shuffle = False
        # currently only gzip compression is supported
        if "_compression" in val:
            compression = val["_compression"]
            shuffle = True

        ds_dtype = (
            fid[val["_dtype"]].dtype
            if val["_dtype"] in fid
            else ("T" if val["_dtype"] == "str" else val["_dtype"])
        )
        if "_vlen" in val:
            ds_name = (
                val["_dtype"].split("_")[0] if "_" in val["_dtype"] else val["_dtype"]
            ) + "_vlen"
            if ds_name not in fid:
                fid[ds_name] = h5py.vlen_dtype(ds_dtype)
                ds_dtype = fid[ds_name]
                fillvalue = None

        return {
            "name": key,
            "shape": ds_shape,
            "maxshape": ds_maxshape,
            "dtype": ds_dtype,
            "chunks": ds_chunk,
            "compression": compression,
            "shuffle": shuffle,
            "fillvalue": fillvalue,
        }

    def __variables(self: H5Create, fid: h5py.File) -> None:
        """Add datasets to HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
           HDF5 file pointer (mode 'w' or 'r+')

        """
        for key, val in self.variables.items():
            # create variable
            is_scalar = False
            if val["_dims"][0] == "scalar":
                is_scalar = True
                dset = fid.create_dataset(**self.__var_scalar(fid, key, val))
            elif val.get("_chunks") == "contiguous":
                dset = fid.create_dataset(**self.__var_nochunk(fid, key, val))
            else:
                dset = fid.create_dataset(**self.__var_chunked(fid, key, val))

            # set attribute FillValue
            if "_FillValue" in val and dset.fillvalue is not None:
                dset.attrs["_FillValue"] = dset.fillvalue

            # add dimension scales
            for ii, coord in enumerate([] if is_scalar else val["_dims"]):
                dset.dims[ii].attach_scale(fid[coord])

            # write data to dataset
            if "_values" in val:
                dset[:] = val["_values"]

            # set all user-suplied attributes
            for attr, attr_val in val.items():
                if attr.startswith("_"):
                    continue
                dset.attrs[attr] = self._adjust_attr(val["_dtype"], attr, attr_val)

    def create(
        self: H5Create,
        filename: Path | str,
        mode: str = "w",
        str_as_bytes: bool = True,
    ) -> None:
        """Create a HDF5/netCDF4 file (overwrite if exist).

        Parameters
        ----------
        filename :  Path | str
           Full name of the HDF5/netCDF4 file to be generated
        mode :  {"r+", "w", "w-", "a"}, default="w"
           The value of mode is passed to h5py.File, see `h5py` documentation
        str_as_bytes: bool, default=True
           Convert string to a netCDF4 compatable byte-array

        """
        self.str2bytes = str_as_bytes
        try:
            with h5py.File(filename, mode, track_order=True, libver=H5_LIBVER) as fid:
                self.__groups(fid)
                self.__dimensions(fid)
                self.__compounds(fid)
                self.__variables(fid)
                self.__attrs(fid)
        except PermissionError as exc:
            raise RuntimeError(f"failed create {filename}") from exc

    def diskless(self: H5Create, str_as_bytes: bool = True) -> h5py.File:
        """Create a HDF5/netCDF4 file in memory.

        Parameters
        ----------
        str_as_bytes: bool, default=True
           Convert string to a netCDF4 compatable byte-array

        Notes
        -----
        An in-memory file will always created from scratch, `mode` has no meaning.

        Returns
        -------
          h5py.File: to add data to the empty HDF5 file

        """
        self.str2bytes = str_as_bytes
        fid = h5py.File.in_memory(track_order=True, libver=H5_LIBVER)
        self.__groups(fid)
        self.__dimensions(fid)
        self.__compounds(fid)
        self.__variables(fid)
        self.__attrs(fid)
        return fid

    def to_disk(self: H5Create, fid: h5py.File, filename: Path | str) -> None:
        """Write in-memory buffer to file.

        Parameters
        ----------
        fid :  h5py.File
           Dataset pointer returned by method `diskless`
        filename :  Path | str
           Name of the file on disk, or file-like object

        """
        fid.flush()
        try:
            with open(filename, "wb") as ff:
                _ = ff.write(fid.id.get_file_image())
        except PermissionError as exc:
            raise RuntimeError(f"failed create {filename}") from exc
