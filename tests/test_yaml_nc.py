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
"""Test module for h5yaml class `NcYaml`."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from h5yaml.yaml_nc import NcYaml


class TestNcYaml:
    """Class to test NcYaml from h5yaml.yaml_nc."""

    NC_DEF = NcYaml(files("h5yaml.Data") / "nc_testing.yaml").nc_def
    FID = NcYaml(files("h5yaml.Data") / "nc_testing.yaml").diskless()

    def test_exceptions(self: TestNcYaml) -> None:
        """Unit-test for the class exdeptions."""
        # tests which should raise an exception because the file can not be created
        l1a_name = "/this/folder/does/not/exists/test.nc"
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            NcYaml(files("h5yaml.Data") / "nc_testing.yaml").create(l1a_name)
        assert f"failed to create {l1a_name}" in str(excinfo)

        l1a_name = Path("/this/folder/does/not/exists/test.nc")
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            NcYaml(files("h5yaml.Data") / "nc_testing.yaml").create(l1a_name)
        assert f"failed to create {l1a_name}" in str(excinfo)

        l1a_name = "/test.nc"
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            NcYaml(files("h5yaml.Data") / "nc_testing.yaml").create(l1a_name)
        assert f"failed to create {l1a_name}" in str(excinfo.value)

        # tests which raise an exception because the YAML file is corrupted
        file_path = Path("nc_testing.yaml")
        with pytest.raises(FileNotFoundError, match=r".* not found") as excinfo:
            _ = NcYaml(file_path).nc_def
        assert f"{file_path} not found" in str(excinfo.value)

    def test_groups(self: TestNcYaml) -> None:
        """Unit-test to check the groups."""
        if "groups" not in self.NC_DEF:
            return

        for key in self.NC_DEF["groups"]:
            pkey = PurePosixPath(key)
            if len(pkey.parts) > 1:
                assert pkey.name in self.FID[pkey.parent].groups
            else:
                assert key in self.FID.groups

    def test_dimensions(self: TestNcYaml) -> None:
        """Unit-test to check the dimensions."""
        if "dimensions" not in self.NC_DEF:
            return

        for key in self.NC_DEF["dimensions"]:
            pkey = PurePosixPath(key)
            dim_grp = self.FID[pkey.parent] if pkey.is_absolute() else self.FID
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

    def test_compounds(self: TestNcYaml) -> None:
        """Unit-test to check the compounds."""
        if "compounds" not in self.NC_DEF:
            return

        for key in self.NC_DEF["compounds"]:
            assert key in self.FID.cmptypes

    def test_variables(self: TestNcYaml) -> None:
        """Unit-test to check the variables."""
        if "variables" not in self.NC_DEF:
            return

        for key in self.NC_DEF["variables"]:
            pkey = PurePosixPath(key)
            var_grp = self.FID[pkey.parent] if pkey.is_absolute() else self.FID
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

    def test_close(self: TestNcYaml) -> None:
        """Close the in-memory netCDF4 file."""
        self.FID.close()
