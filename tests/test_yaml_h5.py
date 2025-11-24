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
"""Test module for h5yaml class `H5Yaml`."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from h5yaml.yaml_h5 import H5Yaml


class TestH5Yaml:
    """Class to test H5Yaml from h5yaml.yaml_h5."""

    H5_DEF = H5Yaml(files("h5yaml.Data") / "h5_testing.yaml").h5_def
    FID = H5Yaml(files("h5yaml.Data") / "h5_testing.yaml").diskless()

    def test_exceptions(self: TestH5Yaml) -> None:
        """..."""
        # tests which should raise an exception because the file can not be created
        l1a_name = "/this/folder/does/not/exists/test.h5"
        with pytest.raises(FileNotFoundError, match=r"[Errno 2] .*") as excinfo:
            _ = H5Yaml(files("h5yaml.Data") / "h5_testing.yaml").create(l1a_name)
        assert "'No such file or directory" in str(excinfo)

        l1a_name = Path("/this/folder/does/not/exists/test.h5")
        with pytest.raises(FileNotFoundError, match=r"[Errno 2] .*") as excinfo:
            _ = H5Yaml(files("h5yaml.Data") / "h5_testing.yaml").create(l1a_name)
        assert "'No such file or directory" in str(excinfo)

        l1a_name = "/test.h5"
        with pytest.raises(RuntimeError, match=r"failed create .*") as excinfo:
            _ = H5Yaml(files("h5yaml.Data") / "h5_testing.yaml").create(l1a_name)
        assert f"failed create {l1a_name}" in str(excinfo.value)

        # tests which raise an exception because the YAML file is corrupted
        file_path = Path("h5_testing.yaml")
        with pytest.raises(FileNotFoundError, match=r".* not found") as excinfo:
            _ = H5Yaml(file_path).h5_def
        assert f"{file_path} not found" in str(excinfo.value)

        with pytest.raises(ValueError, match=r".* unlimited dimension") as excinfo:
            _ = H5Yaml(files("h5yaml.Data") / "h5_unsupported.yaml").create("klad.h5")
        assert "has more than one unlimited dimension" in str(excinfo.value)

    def test_groups(self: TestH5Yaml) -> None:
        """..."""
        if "groups" not in self.H5_DEF:
            return

        for key in self.H5_DEF["groups"]:
            assert key in self.FID

    def test_dimensions(self: TestH5Yaml) -> None:
        """..."""
        if "dimensions" not in self.H5_DEF:
            return

        for key in self.H5_DEF["dimensions"]:
            assert key in self.FID
            if "_dtype" in self.H5_DEF["dimensions"][key]:
                dtype = self.H5_DEF["dimensions"][key]["_dtype"]
                assert self.FID[key].dtype == np.dtype("O") if dtype == "str" else dtype
            if "_size" in self.H5_DEF["dimensions"][key]:
                assert self.FID[key].size == self.H5_DEF["dimensions"][key]["_size"]
            for attr in self.H5_DEF["dimensions"][key]:
                if attr[0] == "_":
                    continue
                assert self.FID[key].attrs[attr] == self.H5_DEF["dimensions"][key][attr]

    def test_compounds(self: TestH5Yaml) -> None:
        """..."""
        if "compounds" not in self.H5_DEF:
            return

        for key in self.H5_DEF["compounds"]:
            assert key in self.FID

    def test_variables(self: TestH5Yaml) -> None:
        """..."""
        if "variables" not in self.H5_DEF:
            return

        dset = self.H5_DEF["variables"]
        for key in dset:
            assert key in self.FID
            # if "_dtype" in self.H5_DEF["variables"][key]:
            #     assert self.FID[key].dtype == self.H5_DEF["variables"][key]["_dtype"]
            if "_size" in dset[key]:
                assert self.FID[key].size == dset[key]["_size"]
            for attr in self.H5_DEF["variables"][key]:
                if attr[0] == "_":
                    continue

                if isinstance(dset[key][attr], list | tuple | np.ndarray):
                    assert np.array_equal(self.FID[key].attrs[attr], dset[key][attr])
                else:
                    assert self.FID[key].attrs[attr] == dset[key][attr]

    def test_close(self: TestH5Yaml) -> None:
        """..."""
        self.FID.close()
