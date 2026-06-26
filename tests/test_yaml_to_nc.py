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
"""Test module for h5yaml class `YamlToNc`."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path, PurePosixPath

import netCDF4
import numpy as np
import pytest

from h5yaml.yaml_to_nc import YamlToNc


class TestYamlToNc:
    """Class to test YamlToNc from h5yaml.yaml_to_nc."""

    _res = YamlToNc(
        [
            files("h5yaml.Data") / "nc_testing.yaml",
            files("h5yaml.Data") / "h5_global_attrs.yaml",
        ]
    )
    NC_DEF = _res.asdict
    FID_NC = _res.diskless()

    def test_exceptions(self: TestYamlToNc) -> None:
        """Unit-test for class exeptions."""
        yaml_path = files("h5yaml.Data") / "h5_testing.yaml"

        # raise an exception because netCDF4 can not have vlen of compound data
        with pytest.raises(ValueError, match=r".*vlen with compounds") as excinfo:
            YamlToNc(yaml_path).diskless()
        assert f"vlen with compounds" in str(excinfo)

        # raise an exception because folder dows not exist (str)
        l1a_name = "/this/folder/does/not/exists/test.nc"
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            YamlToNc(yaml_path).create(l1a_name)
        assert f"failed to create {l1a_name}" in str(excinfo)

        # raise an exception because the file can not be created (Path)
        l1a_name = Path("/this/folder/does/not/exists/test.nc")
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            YamlToNc(yaml_path).create(l1a_name)
        assert f"failed to create {l1a_name}" in str(excinfo)

        # raise exception due to permission error
        l1a_name = "/test.nc"
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            YamlToNc(yaml_path).create(l1a_name)
        assert f"failed to create {l1a_name}" in str(excinfo)

        # raise exception because the YAML file contains errors
        yaml_path = files("h5yaml.Data") / "h5_unsupported.yaml"
        with pytest.raises(ValueError, match=r".* unlimited dimension") as excinfo:
            _ = YamlToNc(yaml_path).diskless()
        assert "more than one unlimited dimension" in str(excinfo)

        # run a sucessful test with method create()
        yaml_path = files("h5yaml.Data") / "nc_testing.yaml"
        l1a_name = Path("tmp_test.nc")
        _ = YamlToNc(yaml_path).create(l1a_name)
        l1a_name.unlink()
        
        # run successful and failing tests with method to_disk()
        res = YamlToNc(yaml_path)
        res.to_disk(res.diskless(), l1a_name)
        l1a_name.unlink()
        # raise exception due to permission error
        l1a_name = Path("/test.nc")
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            res.to_disk(res.diskless(), l1a_name)
        assert f"failed to create {l1a_name}" in str(excinfo)
        # raise an exception because the file can not be created (Path)
        l1a_name = Path("/this/folder/does/not/exists/test.nc")
        with pytest.raises(RuntimeError, match=r"failed to write .*") as excinfo:
            res.to_disk(res.diskless(), l1a_name)
        assert f"failed to write {l1a_name}" in str(excinfo)
        

    def test_nc_groups(self: TestYamlToNc) -> None:
        """Unit-test to check the groups."""
        if "groups" not in self.NC_DEF:
            return

        for key in self.NC_DEF["groups"]:
            assert isinstance(self.FID_NC[key], netCDF4.Group)

    def test_nc_dimensions(self: TestYamlToNc) -> None:
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

    def test_nc_compounds(self: TestYamlToNc) -> None:
        """Unit-test to check the compounds."""
        if "compounds" not in self.NC_DEF:
            return

        for key in self.NC_DEF["compounds"]:
            pkey = PurePosixPath(key)
            if pkey.is_absolute():
                assert pkey.name in self.FID_NC[pkey.parent].cmptypes
            else:
                assert key in self.FID_NC.cmptypes

    def test_nc_variables(self: TestYamlToNc) -> None:
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

    def test_nc_attrs(self: TestYamlToNc) -> None:
        """Unit-test to check the (global) attributes."""
        if "attrs_global" not in self.NC_DEF:
            return

        for key in self.NC_DEF["attrs_global"]:  # .items():
            print(key)

    def test_close(self: TestYamlToNc) -> None:
        """Close the in-memory netCDF4 file."""
        self.FID_NC.close()
