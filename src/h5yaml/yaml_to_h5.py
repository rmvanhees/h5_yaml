#
# This file is part of Python package: `h5yaml`
#
#     https://github.com/rmvanhees/h5_yaml.git
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
"""Generate a structured HDF5 file without data from YAML file using `h5py`."""

from __future__ import annotations

__all__ = ["YamlToH5"]

import logging
from pathlib import Path, PosixPath

import h5py
import numpy as np

from . import sw_version
from .lib.adjust_attr import adjust_attr
from .read_nc_yaml import ReadNcYaml

# - global parameters ---------------------------------
H5_LIBVER = ("v110", "latest")


# - local function -------------------------------------
def str2bytes(ss: str) -> np.bytes_:
    """Convert Python string to ASCII encoded bytes."""
    return np.array(ss, dtype=h5py.string_dtype("ascii", len(ss)))


def find_dimension(fid: h5py.File, var_name: str, dim_name: str) -> h5py.dataset:
    """Find dimension in HDF5 file."""
    for pp in PosixPath(var_name).parents:
        if (dim_path := str(pp / dim_name)) in fid:
            break
    else:
        raise ValueError(f"Dimension '{dim_name}' not found in file")

    return fid[dim_path]


# - class definition -----------------------------------
class YamlToH5(ReadNcYaml):
    """Class to create a structured HDF5 file without data using Python package `h5py`.

    Parameters
    ----------
    nc_yaml_fl :  Path | str | list[Path | str]
       YAML files with the template of a netCDF4/HDF5 file

    """

    str2bytes = False

    def __init__(
        self: YamlToH5,
        nc_yaml_fl: Path | str | list[Path | str],
    ) -> None:
        """Construct a YamlToH5 instance."""
        self.logger = logging.getLogger("h5yaml.YamlToH5")
        super().__init__(nc_yaml_fl)

    def _adjust_attr(
        self: YamlToH5, dtype: str, attr_key: str, attr_val: np.generic
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
        res = adjust_attr(dtype, attr_key, attr_val)
        if self.str2bytes and isinstance(attr_val, str) and attr_val == res:
            return str2bytes(attr_val)

        return res

    def __attrs(self: YamlToH5, fid: h5py.File) -> None:
        """Attach global attributes to file.

        Parameters
        ----------
        fid :  h5py.File
           h5py file pointer (mode 'w' or 'r+')

        """
        value = (
            f"h5yaml.{self.__class__.__name__}(YamlToH5),version={sw_version()}"
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

    def __groups(self: YamlToH5, fid: h5py.File) -> None:
        """Create groups in netCDF4/HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
           h5py file pointer (mode 'w' or 'r+')

        """
        for key in self.groups:
            if key not in fid:
                _ = fid.create_group(key, track_order=True)

        # add group attributes
        for key, value in self.attrs_groups.items():
            if key in fid.attrs or value == "TBW":
                continue
            if isinstance(value, str):
                fid[str(Path(key).parent)].attrs[Path(key).name] = (
                    str2bytes(value) if self.str2bytes else value
                )
            else:
                fid[str(Path(key).parent)].attrs[Path(key).name] = value

    def __dimensions(self: YamlToH5, fid: h5py.File) -> None:
        """Add dimensions to netCDF4/HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
         h5py file pointer (mode 'w' or 'r+')

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

    def __compounds(self: YamlToH5, fid: h5py.File) -> None:
        """Add compound datatypes to netCDF4/HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
           h5py file pointer (mode 'w' or 'r+')

        """
        for key, val in self.compounds.items():
            fid[key] = np.dtype([(k, *v) for k, v in val.items()], align=True)

    def __var_scalar(self: YamlToH5, fid: h5py.File, key: str, val: dict) -> dict:
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

    def __var_nochunk(self: YamlToH5, fid: h5py.File, key: str, val: dict) -> dict:
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
            dim_sz = find_dimension(fid, key, coord).size
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

    def __var_chunked(self: YamlToH5, fid: h5py.File, key: str, val: dict) -> dict:
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
            dim_sz = find_dimension(fid, key, coord).size
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

    def __variables(self: YamlToH5, fid: h5py.File) -> None:
        """Add datasets to netCDF4/HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
           h5py file pointer (mode 'w' or 'r+')

        """
        for key, val in self.variables.items():
            # create variable
            if val["_dims"][0] == "scalar":
                dset = fid.create_dataset(**self.__var_scalar(fid, key, val))
            else:
                if val.get("_chunks") == "contiguous":
                    dset = fid.create_dataset(**self.__var_nochunk(fid, key, val))
                else:
                    dset = fid.create_dataset(**self.__var_chunked(fid, key, val))

                # add dimension scales
                for ii, coord in enumerate(val["_dims"]):
                    dset.dims[ii].attach_scale(find_dimension(fid, key, coord))

            # write data to dataset
            if "_values" in val:
                dset[()] = val["_values"]

            # set attribute _FillValue
            if "_FillValue" in val and dset.fillvalue is not None:
                dset.attrs["_FillValue"] = dset.fillvalue

            # set all user-suplied attributes
            for attr, attr_val in val.items():
                if attr.startswith("_"):
                    continue
                dset.attrs[attr] = self._adjust_attr(val["_dtype"], attr, attr_val)

    def create(
        self: YamlToH5,
        filename: Path | str,
        mode: str = "w",
        str_as_bytes: bool = True,
    ) -> None:
        """Create a structured netCDF4/HDF5 file on disk (overwrite if exist).

        Parameters
        ----------
        filename :  Path | str
           Full name of the netCDF4/HDF5 file to be generated
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
            raise RuntimeError(f"failed to create {filename}") from exc

    def diskless(self: YamlToH5, str_as_bytes: bool = True) -> h5py.File:
        """Create a netCDF4/HDF5 file in memory.

        Parameters
        ----------
        str_as_bytes: bool, default=True
           Convert string to a netCDF4 compatable byte-array

        Notes
        -----
        An in-memory file will always created from scratch, `mode` has no meaning.

        Returns
        -------
          h5py.File: to add data to the structured netCDF4/HDF5 file

        """
        self.str2bytes = str_as_bytes
        fid = h5py.File.in_memory(track_order=True, libver=H5_LIBVER)
        self.__groups(fid)
        self.__dimensions(fid)
        self.__compounds(fid)
        self.__variables(fid)
        self.__attrs(fid)
        return fid

    def to_disk(self: YamlToH5, fid: h5py.File, filename: Path | str) -> None:
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
            raise RuntimeError(f"failed to create {filename}") from exc


# no cover: start
def main() -> None:
    """..."""
    yaml_dir = Path.home() / "git" / "h5_yaml" / "src" / "h5yaml" / "Data"
    aa = YamlToH5(yaml_dir / "h5_testing.yaml")
    aa.to_disk(aa.diskless(), "file_h5_create2.h5")


if __name__ == "__main__":
    main()
# no cover: stop
