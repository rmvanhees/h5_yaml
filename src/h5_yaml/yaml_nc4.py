#
# This file is part of h5_yaml:
#    https://github.com/rmvanhees/h5_yaml.git
#
# Copyright (c) 2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Create HDF5/netCDF4 formatted file from a YAML configuration file using netCDF4."""

from __future__ import annotations

__all__ = ["NcYaml"]

import logging
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from netCDF4 import Dataset

from .conf_from_yaml import conf_from_yaml

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# - global parameters ------------------------------


# - local functions --------------------------------


# - class definition -----------------------------------
class NcYaml:
    """Class to create a HDF5/netCDF4 formated file from a YAML configuration file."""

    def __init__(self: NcYaml, h5_yaml_fl: Path) -> None:
        """Construct a NcYaml instance."""
        self.logger = logging.getLogger("h5_yaml.NcYaml")

        try:
            self._h5_def = conf_from_yaml(h5_yaml_fl)
        except RuntimeError as exc:
            raise RuntimeError from exc

        self.yaml_dir = h5_yaml_fl.parent

    def __groups(self: NcYaml, fid: Dataset) -> None:
        """Create groups in HDF5 product."""
        for key in self.h5_def["groups"]:
            _ = fid.createGroup(key)

    def __dimensions(self: NcYaml, fid: Dataset) -> None:
        """Add dimensions to HDF5 product."""
        for key, value in self.h5_def["dimensions"].items():
            _ = fid.createDimension(key, value["_size"])

            if "long_name" not in val:
                continue

            fillvalue = None
            if "_FillValue" in value:
                fillvalue = (
                    np.nan if value["_FillValue"] == "NaN" else int(value["_FillValue"])
                )

            dset = fid.createVariable(
                key
                value["_dtype"],
                dimensions=(key,),
                fill_value=fillvalue,
                contiguous=False if value["_size"] == 0 else True,
                chunksizes=None,
                compression=None,
            )
            dset.setncattrs(**{k: v for k, v in attrs.items() if not k.startswith("_")})

    def __compounds(self: NcYaml, fid: Dataset) -> dict[str, str | int | float]:
        """Add compound datatypes to HDF5 product."""
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
                "fields": [],
                "units": [],
                "names": [],
            }

            for _key, _val in value.items():
                compounds[key]["dtype"].append((_key, _val[0]))
                compounds[key]["fields"].append(_key)
                if len(_val) == 3:
                    compounds[key]["units"].append(_val[1])
                compounds[key]["names"].append(_val[2] if len(_val) == 3 else _val[1])

            fid[key] = np.dtype(compounds[key]["dtype"])

        return compounds

    def __variables(
        self: NcYaml,
        fid: Dataset,
        compounds: dict[str, str | int | float] | None,
    ) -> None:
        """Add datasets to HDF5 product.

        Parameters
        ----------
        fid :  Dataset
           HDF5 file pointer (mode 'r+')
        compounds :  dict[str, str | int | float]
           Definition of the compound(s) in the product

        """
        for key, val in self.h5_def["variables"].items():
            if val["_dtype"] in fid:
                dtype_dset = fid[val["_dtype"]]
                dtype_size = fid[val["_dtype"]].dtype.itemsize
            else:
                dtype_dset = val["_dtype"]
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

            ds_shape = ()
            ds_maxshape = ()
            for coord in val["_dims"]:
                dim_sz = fid[coord].size
                ds_shape += (dim_sz,)
                ds_maxshape += (dim_sz if dim_sz > 0 else None,)

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
                    chunks=None,
                    maxshape=None,
                    dtype=dtype_dset,
                    fillvalue=fillvalue,
                )
            else:
                if val.get("_vlen"):
                    dtype_dset = h5py.vlen_dtype(dtype_dset)

                print(key, ds_shape, ds_chunk, ds_maxshape, dtype_size)
                dset = fid.create_dataset(
                    key,
                    ds_shape,
                    chunks=ds_chunk if isinstance(ds_chunk, tuple) else tuple(ds_chunk),
                    maxshape=ds_maxshape,
                    dtype=dtype_dset,
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

            if if compounds is not None and val["_dtype"] in compounds:
                dset.attrs["fields"] = compounds[val["_dtype"]]["fields"]
                # dset.attrs["units"] = compounds[val["_dtype"]]["units"]
                dset.attrs["long_name"] = compounds[val["_dtype"]]["names"]

    @property
    def h5_def(self: NcYaml) -> dict:
        """Return definition of the HDF5/netCDF4 product."""
        return self._h5_def

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
                #self.__dimensions(fid)
                #self.__variables(fid, self.__compounds(fid))
        except PermissionError as exc:
            raise RuntimeError(f"failed create {l1a_name}") from exc

# - test module -------------------------
def tests() -> None:
    """..."""
    nc_def = conf_from_yaml(files("h5_yaml.Data") / "h5_testing.yaml")
    NcYaml(nc_def).create("test_yaml_nc4.nc")


if __name__ == "__main__":
    tests()
