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
"""Test module for h5yaml class `H5Yaml`."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from h5yaml.nc_from_yaml import NcFromYaml


class TestFromYaml:
    """Class to test H5Yaml from h5yaml.nc_from_yaml_h5."""

    H5_DEF = NcFromYaml(files("h5yaml.Data") / "h5_testing.yaml").asdict
    FID_H5 = NcFromYaml(files("h5yaml.Data") / "h5_testing.yaml").diskless(
        module="h5py"
    )
    NC_DEF = NcFromYaml(files("h5yaml.Data") / "nc_testing.yaml").asdict
    FID_NC = NcFromYaml(files("h5yaml.Data") / "nc_testing.yaml").diskless(
        module="netCDF4"
    )

    def test_exceptions(self: TestFromYaml) -> None:
        """Unit-test for class exeptions."""
        yaml_path = files("h5yaml.Data") / "h5_testing.yaml"
        # raise an exception because folder dows not exist (str)
        l1a_name = "/this/folder/does/not/exists/test.h5"
        with pytest.raises(FileNotFoundError, match=r"[Errno 2] .*") as excinfo:
            NcFromYaml(yaml_path).create(l1a_name, module="h5py")
        assert "'No such file or directory" in str(excinfo)
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            NcFromYaml(yaml_path).create(l1a_name, module="netCDF4")
        assert f"failed to create {l1a_name}" in str(excinfo)

        # raise an exception because the file can not be created (Path)
        l1a_name = Path("/this/folder/does/not/exists/test.h5")
        with pytest.raises(FileNotFoundError, match=r"[Errno 2] .*") as excinfo:
            NcFromYaml(yaml_path).create(l1a_name, module="h5py")
        assert "'No such file or directory" in str(excinfo)
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            NcFromYaml(yaml_path).create(l1a_name, module="netCDF4")
        assert f"failed to create {l1a_name}" in str(excinfo)

        # raise exception due to permission error
        l1a_name = "/test.h5"
        with pytest.raises(RuntimeError, match=r"failed create .*") as excinfo:
            NcFromYaml(yaml_path).create(l1a_name, module="h5py")
        assert f"failed create {l1a_name}" in str(excinfo.value)
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            NcFromYaml(yaml_path).create(l1a_name, module="netCDF4")
        assert f"failed to create {l1a_name}" in str(excinfo)

        # raise exception because YAML file can not be found
        yaml_path = Path("h5_testing.yaml")
        with pytest.raises(FileNotFoundError, match=r".* not found") as excinfo:
            _ = NcFromYaml(yaml_path).asdict
        assert f"{yaml_path} not found" in str(excinfo.value)

        # raise exception because the YAML file contains errors
        yaml_path = files("h5yaml.Data") / "h5_unsupported.yaml"
        with pytest.raises(ValueError, match=r".* unlimited dimension") as excinfo:
            NcFromYaml(yaml_path).diskless(module="h5py")
        assert "has more than one unlimited dimension" in str(excinfo.value)
        with pytest.raises(ValueError, match=r".* unlimited dimension") as excinfo:
            NcFromYaml(yaml_path).create("klad.nc", module="netCDF4")
        assert "more than one unlimited dimension" in str(excinfo)
        if Path("klad.nc").is_file():
            Path("klad.nc").unlink()

    def test_h5_groups(self: TestFromYaml) -> None:
        """Unit-test to check the groups."""
        if "groups" not in self.H5_DEF:
            return

        for key in self.H5_DEF["groups"]:
            assert key in self.FID_H5

    def test_nc_groups(self: TestFromYaml) -> None:
        """Unit-test to check the groups."""
        if "groups" not in self.NC_DEF:
            return

        for key in self.NC_DEF["groups"]:
            pkey = PurePosixPath(key)
            if len(pkey.parts) > 1:
                assert pkey.name in self.FID_NC[pkey.parent].groups
            else:
                assert key in self.FID_NC.groups

    def test_h5_dimensions(self: TestFromYaml) -> None:
        """Unit-test to check the dimensions."""
        if "dimensions" not in self.H5_DEF:
            return

        for key in self.H5_DEF["dimensions"]:
            assert key in self.FID_H5
            if "_dtype" in self.H5_DEF["dimensions"][key]:
                dtype = self.H5_DEF["dimensions"][key]["_dtype"]
                assert (
                    self.FID_H5[key].dtype == np.dtype("O") if dtype == "str" else dtype
                )
            if "_size" in self.H5_DEF["dimensions"][key]:
                assert self.FID_H5[key].size == self.H5_DEF["dimensions"][key]["_size"]
            if "_values" in self.H5_DEF["dimensions"][key]:
                assert np.array_equal(
                    [x.decode() for x in self.FID_H5[key][:]],
                    self.H5_DEF["dimensions"][key]["_values"],
                )
            if "_range" in self.H5_DEF["dimensions"][key]:
                assert np.array_equal(
                    self.FID_H5[key][:],
                    np.arange(*self.H5_DEF["dimensions"][key]["_range"]),
                )
            for attr in self.H5_DEF["dimensions"][key]:
                if attr[0] == "_":
                    continue
                assert (
                    self.FID_H5[key].attrs[attr] == self.H5_DEF["dimensions"][key][attr]
                )

    def test_nc_dimensions(self: TestFromYaml) -> None:
        """Unit-test to check the dimensions."""
        if "dimensions" not in self.NC_DEF:
            return

        for key in self.NC_DEF["dimensions"]:
            pkey = PurePosixPath(key)
            dim_grp = self.FID_NC[pkey.parent] if pkey.is_absolute() else self.FID_NC
            dim_name = pkey.name if pkey.is_absolute() else key
            assert dim_name in dim_grp.dimensions

            if dim_name in dim_grp.variables:
                nc_dim = dim_grp.variables[dim_name]
                if "_dtype" in self.NC_DEF["dimensions"][key]:
                    dtype = self.NC_DEF["dimensions"][key]["_dtype"]
                    assert nc_dim.dtype == str if dtype == "str" else dtype
                if "_size" in self.NC_DEF["dimensions"][key]:
                    assert nc_dim.size == self.NC_DEF["dimensions"][key]["_size"]
                if "_values" in self.NC_DEF["dimensions"][key]:
                    assert np.array_equal(
                        nc_dim[:], self.NC_DEF["dimensions"][key]["_values"]
                    )
                if "_range" in self.NC_DEF["dimensions"][key]:
                    assert np.array_equal(
                        nc_dim[:], np.arange(*self.NC_DEF["dimensions"][key]["_range"])
                    )
                for attr in self.NC_DEF["dimensions"][key]:
                    if attr[0] == "_":
                        continue

                    assert getattr(nc_dim, attr) == self.NC_DEF["dimensions"][key][attr]

    def test_h5_compounds(self: TestFromYaml) -> None:
        """Unit-test to check the compounds."""
        if "compounds" not in self.H5_DEF:
            return

        for key in self.H5_DEF["compounds"]:
            assert key in self.FID_H5

    def test_nc_compounds(self: TestFromYaml) -> None:
        """Unit-test to check the compounds."""
        if "compounds" not in self.NC_DEF:
            return

        for key in self.NC_DEF["compounds"]:
            assert key in self.FID_NC.cmptypes

    def test_h5_variables(self: TestFromYaml) -> None:
        """Unit-test to check the variables."""
        if "variables" not in self.H5_DEF:
            return

        dset = self.H5_DEF["variables"]
        for key in dset:
            assert key in self.FID_H5
            # if "_dtype" in self.H5_DEF["variables"][key]:
            #     assert (
            #        self.FID_H5[key].dtype == self.H5_DEF["variables"][key]["_dtype"]
            #     )
            if "_size" in dset[key]:
                assert self.FID_H5[key].size == dset[key]["_size"]
            for attr in self.H5_DEF["variables"][key]:
                if attr[0] == "_":
                    continue

                if isinstance(dset[key][attr], list | tuple | np.ndarray):
                    assert np.array_equal(self.FID_H5[key].attrs[attr], dset[key][attr])
                else:
                    assert self.FID_H5[key].attrs[attr] == dset[key][attr]

    def test_nc_variables(self: TestFromYaml) -> None:
        """Unit-test to check the variables."""
        if "variables" not in self.NC_DEF:
            return

        for key in self.NC_DEF["variables"]:
            pkey = PurePosixPath(key)
            var_grp = self.FID_NC[pkey.parent] if pkey.is_absolute() else self.FID_NC
            var_name = pkey.name if pkey.is_absolute() else key
            assert var_name in var_grp.variables
            nc_var = var_grp.variables[var_name]

            if "_size" in self.NC_DEF["variables"][key]:
                assert nc_var.size == self.NC_DEF["variables"][key]["_size"]

            for attr in self.NC_DEF["variables"][key]:
                if attr[0] == "_":
                    continue

                if isinstance(
                    self.NC_DEF["variables"][key][attr], list | tuple | np.ndarray
                ):
                    assert np.array_equal(
                        getattr(nc_var, attr), self.NC_DEF["variables"][key][attr]
                    )
                else:
                    assert getattr(nc_var, attr) == self.NC_DEF["variables"][key][attr]

    def test_close(self: TestFromYaml) -> None:
        """Close the in-memory HDF5 file."""
        self.FID_H5.close()
        self.FID_NC.close()
