#
# This file is part of h5_yaml
#    https://github.com/rmvanhees/h5_yaml.git
#
# Copyright (c) 2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Create HDF5/netCDF4 formatted file from a YAML configuration file using h5py."""

from __future__ import annotations

__all__ = ["H5Yaml"]

import logging
from importlib.resources import files
from pathlib import Path

import h5py
import numpy as np

from h5yaml.conf_from_yaml import conf_from_yaml
from h5yaml.lib.chunksizes import guess_chunks


# - class definition -----------------------------------
class H5Yaml:
    """Class to create a HDF5/netCDF4 formated file from a YAML configuration file.

    Parameters
    ----------
    h5_yaml_fl :  Path
       YAML files with the HDF5 format definition

    """

    def __init__(self: H5Yaml, h5_yaml_fl: Path) -> None:
        """Construct a H5Yaml instance."""
        self.logger = logging.getLogger("h5yaml.H5Yaml")

        try:
            self._h5_def = conf_from_yaml(h5_yaml_fl)
        except RuntimeError as exc:
            raise RuntimeError from exc

        self.yaml_dir = h5_yaml_fl.parent

    def __groups(self: H5Yaml, fid: h5py.File) -> None:
        """Create groups in HDF5 product."""
        for key in self.h5_def["groups"]:
            _ = fid.create_group(key)

    def __dimensions(self: H5Yaml, fid: h5py.File) -> None:
        """Add dimensions to HDF5 product."""
        for key, value in self.h5_def["dimensions"].items():
            fillvalue = None
            if "_FillValue" in value:
                fillvalue = (
                    np.nan if value["_FillValue"] == "NaN" else int(value["_FillValue"])
                )

            if value["_size"] == 0:
                ds_chunk = value.get("_chunks", (50,))
                dset = fid.create_dataset(
                    key,
                    shape=(0,),
                    dtype=(
                        h5py.string_dtype()
                        if value["_dtype"] == "str"
                        else value["_dtype"]
                    ),
                    chunks=ds_chunk if isinstance(ds_chunk, tuple) else tuple(ds_chunk),
                    maxshape=(None,),
                    fillvalue=fillvalue,
                )
            else:
                dset = fid.create_dataset(
                    key,
                    shape=(value["_size"],),
                    dtype=value["_dtype"],
                )
                if "_values" in value:
                    dset[:] = value["_values"]

            dset.make_scale(
                Path(key).name
                if "long_name" in value
                else "This is a netCDF dimension but not a netCDF variable."
            )
            for attr, attr_val in value.items():
                if attr.startswith("_"):
                    continue
                dset.attrs[attr] = attr_val

    def __compounds(self: H5Yaml, fid: h5py.File) -> dict[str, str | int | float]:
        """Add compound datatypes to HDF5 product."""
        if "compounds" not in self.h5_def:
            return {}

        compounds = {}
        if isinstance(self.h5_def["compounds"], list):
            file_list = self.h5_def["compounds"].copy()
            self.h5_def["compounds"] = {}
            for name in file_list:
                if not (yaml_fl := self.yaml_dir / name).is_file():
                    continue
                try:
                    res = conf_from_yaml(yaml_fl)
                except RuntimeError as exc:
                    raise RuntimeError from exc
                for key, value in res.items():
                    self.h5_def["compounds"][key] = value

        for key, value in self.h5_def["compounds"].items():
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

            fid[key] = np.dtype(compounds[key]["dtype"])

        return compounds

    def __variables(
        self: H5Yaml, fid: h5py.File, compounds: dict[str, str | int | float] | None
    ) -> None:
        """Add datasets to HDF5 product.

        Parameters
        ----------
        fid :  h5py.File
           HDF5 file pointer (mode 'r+')
        compounds :  dict[str, str | int | float]
           Definition of the compound(s) in the product

        """
        for key, val in self.h5_def["variables"].items():
            if val["_dtype"] in fid:
                ds_dtype = fid[val["_dtype"]]
                dtype_size = fid[val["_dtype"]].dtype.itemsize
            else:
                ds_dtype = val["_dtype"]
                dtype_size = np.dtype(val["_dtype"]).itemsize

            fillvalue = None
            if "_FillValue" in val:
                fillvalue = (
                    np.nan if val["_FillValue"] == "NaN" else int(val["_FillValue"])
                )

            compression = None
            shuffle = False
            # currently only gzip compression is supported
            if "_compression" in val:
                compression = val["_compression"]
                shuffle = True

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
                raise ValueError("more than one unlimited dimension")

            # obtain chunk-size settings
            ds_chunk = (
                val["_chunks"]
                if "_chunks" in val
                else guess_chunks(ds_shape, dtype_size)
            )

            # create the variable
            if ds_chunk == "contiguous":
                dset = fid.create_dataset(
                    key,
                    ds_shape,
                    dtype=ds_dtype,
                    chunks=None,
                    maxshape=None,
                    fillvalue=fillvalue,
                )
            else:
                if val.get("_vlen"):
                    ds_dtype = h5py.vlen_dtype(ds_dtype)
                    fillvalue = None
                    if ds_maxshape == (None,):
                        ds_chunk = (16,)

                dset = fid.create_dataset(
                    key,
                    ds_shape,
                    dtype=ds_dtype,
                    chunks=ds_chunk if isinstance(ds_chunk, tuple) else tuple(ds_chunk),
                    maxshape=ds_maxshape,
                    fillvalue=fillvalue,
                    compression=compression,
                    shuffle=shuffle,
                )

            for ii, coord in enumerate(val["_dims"]):
                dset.dims[ii].attach_scale(fid[coord])

            for attr, attr_val in val.items():
                if attr.startswith("_"):
                    continue
                dset.attrs[attr] = attr_val

            if compounds is not None and val["_dtype"] in compounds:
                if compounds[val["_dtype"]]["units"]:
                    dset.attrs["units"] = compounds[val["_dtype"]]["units"]
                if compounds[val["_dtype"]]["names"]:
                    dset.attrs["long_name"] = compounds[val["_dtype"]]["names"]

    @property
    def h5_def(self: H5Yaml) -> dict:
        """Return definition of the HDF5/netCDF4 product."""
        return self._h5_def

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
                self.__variables(fid, self.__compounds(fid))
        except PermissionError as exc:
            raise RuntimeError(f"failed create {l1a_name}") from exc


# - test module -------------------------
def tests() -> None:
    """..."""
    print("Calling H5Yaml")
    H5Yaml(files("h5yaml.Data") / "h5_testing.yaml").create("test_yaml.h5")


if __name__ == "__main__":
    tests()
