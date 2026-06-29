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
"""Test module for h5yaml class `TemplateH5` using package h5py."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from h5yaml.template_h5 import TemplateH5


class TestTemplateH5:
    """Class to test TemplateH5 from h5yaml.yaml_to_h5."""

    _res = TemplateH5(
        [
            files("h5yaml.Data") / "h5_testing.yaml",
            files("h5yaml.Data") / "h5_global_attrs.yaml",
        ]
    )
    H5_DEF = _res.asdict
    FID_H5 = _res.diskless(str_as_bytes=False)

    def test_exceptions(self: TestTemplateH5) -> None:
        """Unit-test for class exeptions."""
        yaml_path = files("h5yaml.Data") / "h5_testing.yaml"
        # raise an exception because folder does not exist (str)
        l1a_name = "/this/folder/does/not/exists/test.h5"
        with pytest.raises(FileNotFoundError, match=r"[Errno 2] .*") as excinfo:
            TemplateH5(yaml_path).create(l1a_name)
        assert "FileNotFoundError" in str(excinfo)

        # raise an exception because the file can not be created (Path)
        l1a_name = Path("/this/folder/does/not/exists/test.h5")
        with pytest.raises(FileNotFoundError, match=r"[Errno 2] .*") as excinfo:
            TemplateH5(yaml_path).create(l1a_name)
        assert "'No such file or directory" in str(excinfo)

        # raise exception due to permission error
        l1a_name = "/test.h5"
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            TemplateH5(yaml_path).create(l1a_name)
        assert f"failed to create {l1a_name}" in str(excinfo.value)

        # raise exception because the YAML file contains errors
        yaml_path = files("h5yaml.Data") / "h5_unsupported.yaml"
        with pytest.raises(ValueError, match=r".* unlimited dimension") as excinfo:
            _ = TemplateH5(yaml_path).diskless()
        assert "has more than one unlimited dimension" in str(excinfo.value)

        # run a sucessful test with method create()
        yaml_path = files("h5yaml.Data") / "h5_testing.yaml"
        l1a_name = Path("tmp_test.h5")
        TemplateH5(yaml_path).create(l1a_name)
        l1a_name.unlink()

        # run successful and failing tests with method to_disk()
        res = TemplateH5(yaml_path)
        res.to_disk(res.diskless(), l1a_name)
        l1a_name.unlink()
        # raise exception due to permission error
        l1a_name = Path("/test.h5")
        with pytest.raises(RuntimeError, match=r"failed to create .*") as excinfo:
            res.to_disk(res.diskless(), l1a_name)
        assert f"failed to create {l1a_name}" in str(excinfo)
        # raise an exception because path to file does not exist
        l1a_name = Path("/this/folder/does/not/exists/test.h5")
        with pytest.raises(FileNotFoundError, match=r"[Errno 2] .*") as excinfo:
            res.to_disk(res.diskless(), l1a_name)
        assert "No such file or directory" in str(excinfo)

    def test_h5_groups(self: TestTemplateH5) -> None:
        """Unit-test to check the groups."""
        if "groups" not in self.H5_DEF:
            return

        for key in self.H5_DEF["groups"]:
            assert key in self.FID_H5

    def test_h5_dimensions(self: TestTemplateH5) -> None:
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

    def test_h5_compounds(self: TestTemplateH5) -> None:
        """Unit-test to check the compounds."""
        if "compounds" not in self.H5_DEF:
            return

        for key in self.H5_DEF["compounds"]:
            assert key in self.FID_H5

    def test_h5_variables(self: TestTemplateH5) -> None:
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

    def test_nc_attrs(self: TestTemplateH5) -> None:
        """Unit-test to check the (global) attributes."""
        if "attrs_global" not in self.H5_DEF:
            return

        for key in self.H5_DEF["attrs_global"]:  # .items():
            print(key)

    def test_close(self: TestTemplateH5) -> None:
        """Close the in-memory HDF5 file."""
        self.FID_H5.close()
