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
"""Generate a template netCDF4 file which contains all mandatory netCDF4 elements.

The template file may contain any of the following netCDF4 elements:

- "groups" to organize the data in hierarchical groups
- "dimensions" which define the sizes of all variables in terms of dimensions
- "variables" which behave as much like `numpy` arrays
- "attributes" which can be attached to the file of one of its groups

"""

from __future__ import annotations

__all__ = ["TemplateNc"]

import logging
from pathlib import Path, PurePosixPath

import numpy as np

# pylint: disable=no-name-in-module
from netCDF4 import Dataset

from . import sw_version
from .lib.adjust_attr import adjust_attr
from .template import Template


# - local function -------------------------------------
def get_dim_size(fid: Dataset, pkey: PurePosixPath, coord: PurePosixPath) -> int:
    """Get size of variable dimension.

    Parameters
    ----------
    pkey :  PurePosixPath
       Full path to the variable
    coord :  PurePosixPath
       Name of the dimension or its full path

    """
    # check if dimension in same group as variable
    if len(pkey.parents) == len(coord.parents):
        if len(coord.parents) == 1:
            return fid.dimensions[coord.name].size

        return fid[coord.parent].dimensions[coord.name].size

    # search dimension on its name
    dim_name = coord.name
    for pp in pkey.parents:
        if pp == PurePosixPath("/") and dim_name in fid.dimensions:
            return fid.dimensions[dim_name].size
        if dim_name in fid[pp].dimensions:
            return fid[pp].dimensions[dim_name].size

    raise ValueError(f"Dimension '{coord}' not found in file")


def get_cmp_dtype(fid: Dataset, compound: dict, val: dict) -> np.dtype:
    """Get netCDF4 compound data-type."""
    cmp_t = np.dtype([(k, *v) for k, v in compound.items()])
    pkey = PurePosixPath(val["_dtype"])
    if pkey.is_absolute():
        return (
            fid[pkey.parent].cmptypes[pkey.name]
            if pkey.name in fid[pkey.parent].cmptypes
            else fid[pkey.parent].createCompoundType(cmp_t, pkey.name)
        )

    return (
        fid.cmptypes[val["_dtype"]]
        if val["_dtype"] in fid.cmptypes
        else fid.createCompoundType(cmp_t, val["_dtype"])
    )


# - class definition -----------------------------------
class TemplateNc(Template):
    """Class to create a template netCDF4 file using Python package `netCDF4`.

    Parameters
    ----------
    nc_yaml :  list[Path | str] | Path | str | None, default=None
       YAML files with the template netCDF4 file definition

    nc_dict :  dict[str, dict] | None, default=None
       Python dictionary with the template netCDF4 file definition

    """

    def __init__(
        self: TemplateNc,
        nc_yaml: Path | str | list[Path | str] | None = None,
        *,
        nc_dict: dict | None = None,
    ) -> None:
        """Construct a TemplateNc instance."""
        self.logger = logging.getLogger("h5yaml.TemplateNc")
        super().__init__(nc_yaml, nc_dict)

    def __attrs(self: TemplateNc, fid: Dataset) -> None:
        """Attach global attributes to file.

        Parameters
        ----------
        fid :  netCDF4.Dataset
           netCDF4 Dataset (mode 'r+')

        """
        fid.setncattr(
            "_NCCreator",
            (f"h5yaml.{self.__class__.__name__}(Template),version={sw_version()}"),
        )
        for key, val in self.attrs_global.items():
            if val == "TBW":
                continue

            fid.setncattr(key, val)

    def __groups(self: TemplateNc, fid: Dataset) -> None:
        """Create groups in a netCDF4 product.

        Parameters
        ----------
        fid :  netCDF4.Dataset
           netCDF4 Dataset (mode 'r+')

        """
        for key in self.groups:
            _ = fid.createGroup(key)

        # add group attributes
        for key, val in self.attrs_groups.items():
            if val == "TBW":
                continue

            fid[str(Path(key).parent)].setncattr(Path(key).name, val)

    def __dimensions(self: TemplateNc, fid: Dataset) -> None:
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
                    dset[:] = np.array(value["_values"], dtype=value["_dtype"])
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
        self: TemplateNc, fid: Dataset, key: str, val: dict, compound: None | dict
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

        return {
            "varname": key,
            "datatype": (
                val["_dtype"] if compound is None else get_cmp_dtype(fid, compound, val)
            ),
            "fill_value": fillvalue,
        }

    def __var_nochunk(
        self: TemplateNc, fid: Dataset, key: str, val: dict, compound: None | dict
    ) -> dict:
        """Return parameters to create a variable without chunking.

        Parameters
        ----------
        fid :  Dataset
           netCDF4 file-pointer
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

        return {
            "varname": key,
            "datatype": (
                val["_dtype"] if compound is None else get_cmp_dtype(fid, compound, val)
            ),
            "dimensions": [PurePosixPath(x).name for x in val["_dims"]],
            "contiguous": True,
            "fill_value": fillvalue,
        }

    def __var_chunked(
        self: TemplateNc,
        fid: Dataset,
        pkey: PurePosixPath,
        val: dict,
        compound: None | dict,
    ) -> dict:
        """Return parameters to create a variable with chunking.

        Parameters
        ----------
        fid :  Dataset
           netCDF4 file-pointer
        pkey :  PurePosixPath
           Variable name (full-path)
        val :  dict
           Properties of the variable
        compound :  None | dict
           Properties of the compound data-type

        """
        fillvalue = None
        if compound is None and "_FillValue" in val:
            fillvalue = np.nan if val["_FillValue"] == "NaN" else int(val["_FillValue"])

        compression = None
        complevel = 0
        # currently only gzip compression is supported
        if "_compression" in val:
            compression = "zlib"
            complevel = val["_compression"]

        ds_chunk = val.get("_chunks")
        if ds_chunk is not None and not isinstance(ds_chunk, bool):
            if -1 in ds_chunk:
                ii = ds_chunk.index(-1)
                ds_chunk[ii] = int(
                    get_dim_size(fid, pkey, PurePosixPath(val["_dims"][ii]))
                )
            ds_chunk = tuple(ds_chunk)

        datatype = (
            val["_dtype"] if compound is None else get_cmp_dtype(fid, compound, val)
        )
        if "_vlen" in val:
            if val["_dtype"] in fid.cmptypes:
                raise ValueError("can not have vlen with compounds")
            datatype = fid.createVLType(datatype, "phony_vlen")
            fillvalue = None

        return {
            "varname": pkey.name,
            "datatype": datatype,
            "dimensions": [PurePosixPath(x).name for x in val["_dims"]],
            "fill_value": fillvalue,
            "compression": compression,
            "complevel": complevel,
            "chunksizes": ds_chunk,
        }

    def __variables(self: TemplateNc, fid: Dataset) -> None:
        """Add datasets to a netCDF4 product.

        Parameters
        ----------
        fid :  netCDF4.Dataset
           netCDF4 Dataset (mode 'r+')

        """
        for key, val in self.variables.items():
            pkey = PurePosixPath(key)
            var_grp = fid[pkey.parent] if len(pkey.parents) > 1 else fid

            # check if dtype of variable is a compound
            compound = self.compounds.get(val["_dtype"], None)

            # create variable
            if val["_dims"][0] == "scalar":
                dset = var_grp.createVariable(
                    **self.__var_scalar(fid, pkey.name, val, compound)
                )
                # write data to variable (works currently only for scalar datasets)
                if "_values" in val:
                    dset[:] = val["_values"]
            else:
                # check number of unlimited dimensions
                n_udim = 0
                for coord in val["_dims"]:
                    n_udim += int(get_dim_size(fid, pkey, PurePosixPath(coord)) == 0)

                # currently, we can not handle more than one unlimited dimension
                if n_udim > 1:
                    raise ValueError(f"{key} has more than one unlimited dimension")

                if val.get("_chunks") == "contiguous":
                    dset = var_grp.createVariable(
                        **self.__var_nochunk(fid, pkey.name, val, compound)
                    )
                else:
                    dset = var_grp.createVariable(
                        **self.__var_chunked(fid, pkey, val, compound)
                    )

            # add user-supplied attributes
            dset.setncatts(
                {
                    k: adjust_attr(val["_dtype"], k, v)
                    for k, v in val.items()
                    if not k.startswith("_")
                }
            )

    def create(self: TemplateNc, filename: Path | str, mode: str = "w") -> None:
        """Create a structured netCDF4 file on disk (overwrite if exist).

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

    def diskless(self: TemplateNc) -> Dataset:
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

    def to_disk(self: TemplateNc, fid: Dataset, filename: Path | str) -> None:
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
            raise RuntimeError(f"failed to create {filename}") from exc
        except OSError as exc:
            raise RuntimeError(f"failed to write {filename}") from exc


def main() -> None:  # no cover: start
    """..."""
    yaml_dir = Path.home() / "git" / "h5_yaml" / "src" / "h5yaml" / "Data"
    aa = TemplateNc(yaml_dir / "nc_testing.yaml")

    aa.to_disk(aa.diskless(), "file_nc_create2.nc")


if __name__ == "__main__":
    main()  # no cover: stop
