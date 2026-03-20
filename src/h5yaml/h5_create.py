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

from . import __version__

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
    str_as_bytes: bool = False

    """

    def __init__(
        self: H5Create,
        groups: set | None = None,
        compounds: dict | None = None,
        dimensions: dict | None = None,
        variables: dict | None = None,
        attrs_global: dict | None = None,
        attrs_groups: dict | None = None,
        str_as_bytes: bool = False,
    ) -> None:
        """Construct a H5Create instance."""
        self.logger = logging.getLogger("h5yaml.H5Create")
        self.groups = set() if groups is None else groups
        self.compounds = {} if compounds is None else compounds
        self.dimensions = {} if dimensions is None else dimensions
        self.variables = {} if variables is None else variables
        self.attrs_global = {} if attrs_global is None else attrs_global
        self.attrs_groups = {} if attrs_groups is None else attrs_groups
        self.str_as_bytes = str_as_bytes

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
            return eval(attr_val)

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

        if self.str_as_bytes and isinstance(attr_val, str):
            return str2bytes(attr_val)

        return attr_val

    def __attrs(self: H5Create, fid: h5py.File) -> None:
        """Create global and group attributes.

        Parameters
        ----------
        fid :  h5py.File
           HDF5 file pointer (mode 'w' or 'r+')

        """
        fid.attrs["_NCCreator"] = str2bytes(
            f"h5yaml.{self.__class__.__name__}(H5Create)"
            f",version={__version__.split('+', maxsplit=1)[0]}"
        )
        fid.attrs["_NCProperties"] = str2bytes(
            f"version=2,hdf5={h5py.version.hdf5_version}"
        )
        for key, value in self.attrs_global.items():
            if key not in fid.attrs and value != "TBW":
                fid.attrs[key] = str2bytes(value) if isinstance(value, str) else value

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
            if key not in fid.attrs and value != "TBW":
                fid[str(Path(key).parent)].attrs[Path(key).name] = (
                    str2bytes(value) if isinstance(value, str) else value
                )

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

            dset.make_scale(
                Path(key).name
                if "long_name" in val
                else "This is a netCDF dimension but not a netCDF variable."
            )
            if fillvalue is not None:
                dset.attrs["_FillValue"] = dset.fillvalue

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
            fid[key] = np.dtype([(k, v[0]) for k, v in val.items()], align=True)

    def __variables(self: H5Create, fid: h5py.File) -> None:
        """Add datasets to HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
           HDF5 file pointer (mode 'w' or 'r+')

        """

        def add_compound_attr() -> None:
            compound = self.compounds[val["_dtype"]]
            res = [v[2] for k, v in compound.items() if len(v) == 3]
            if res:
                dset.attrs["units"] = [v[1] for k, v in compound.items()]
                dset.attrs["names"] = res
            else:
                dset.attrs["names"] = [v[1] for k, v in compound.items()]

        for key, val in self.variables.items():
            if val["_dtype"] in fid:  # True when compound-dtype
                is_compound = True
                ds_dtype = fid[val["_dtype"]].dtype
            else:
                is_compound = False
                ds_dtype = "T" if val["_dtype"] == "str" else val["_dtype"]

            fillvalue = None
            if "_FillValue" in val:
                fillvalue = np.nan if val["_FillValue"] == "NaN" else val["_FillValue"]

            # check for scalar dataset
            if val["_dims"][0] == "scalar":
                dset = fid.create_dataset(key, (), dtype=ds_dtype, fillvalue=fillvalue)
                if fillvalue is not None:
                    dset.attrs["_FillValue"] = dset.fillvalue
                for attr, attr_val in val.items():
                    if attr.startswith("_"):
                        continue
                    dset.attrs[attr] = self._adjust_attr(val["_dtype"], attr, attr_val)

                if is_compound:
                    add_compound_attr()
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
                    maxshape=None,
                    fillvalue=fillvalue,
                )
                if fillvalue is not None:
                    dset.attrs["_FillValue"] = dset.fillvalue
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
                if fillvalue is not None:
                    dset.attrs["_FillValue"] = dset.fillvalue

            for ii, coord in enumerate(val["_dims"]):
                dset.dims[ii].attach_scale(fid[coord])

            for attr, attr_val in val.items():
                if not attr.startswith("_"):
                    dset.attrs[attr] = self._adjust_attr(val["_dtype"], attr, attr_val)

            if is_compound:
                add_compound_attr()

    def create(self: H5Create, filename: Path | str) -> None:
        """Create a HDF5/netCDF4 file (overwrite if exist).

        Parameters
        ----------
        filename :  Path | str
           Full name of the HDF5/netCDF4 file to be generated

        """
        try:
            with h5py.File(filename, "w", track_order=True, libver=H5_LIBVER) as fid:
                self.__groups(fid)
                self.__dimensions(fid)
                self.__compounds(fid)
                self.__variables(fid)
                self.__attrs(fid)
        except PermissionError as exc:
            raise RuntimeError(f"failed create {filename}") from exc

    def diskless(self: H5Create) -> h5py.File:
        """Create a HDF5/netCDF4 file in memory."""
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
