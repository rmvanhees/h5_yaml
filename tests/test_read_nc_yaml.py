#
# This file is part of Python package: `h5yaml`
#
#     https://github.com/rmvanhees/h5_yaml.git
#
# Copyright (c) 2026 - R.M. van Hees (SRON)
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
"""Test module for h5yaml class `ReadNcYaml`."""

from __future__ import annotations

from importlib.resources import files

import pytest

from h5yaml.read_nc_yaml import ReadNcYaml


class TestReadNcYaml:
    """Class to test ReadNcYaml from h5yaml.read_nc_yaml."""

    def test_exceptions(self: TestReadNcYaml) -> None:
        """Unit-test for class exeptions."""
        res = ReadNcYaml(
            [
                files("h5yaml.Data") / "h5_testing.yaml",
                files("h5yaml.Data") / "h5_global_attrs.yaml",
            ]
        )
        res = ReadNcYaml(files("h5yaml.Data") / "nc_testing.yaml")
        res = ReadNcYaml(str(files("h5yaml.Data") / "h5_unsupported.yaml"))
        assert isinstance(repr(res), str)

        # raise exception because YAML file can not be found
        with pytest.raises((FileNotFoundError, RuntimeError)) as excinfo:
            res = ReadNcYaml(files("h5yaml.Data") / "h5_testing2.yaml")
        assert "Fails to access" in str(excinfo)

    def test_set_dims(self: TestReadNcYaml) -> None:
        """Unit-test when unlimited dimensions are fixed."""
        res = ReadNcYaml([files("h5yaml.Data") / "nc_testing.yaml"])
        if not ("time" in res.dimensions and res.dimensions["time"]["_size"] <= 0):
            raise ValueError("Can not test method set_dims()")
        res.set_dims({"time": 128})
        assert res.dimensions["time"]["_size"] == 128

        with pytest.raises(KeyError, match=r"unknown_dim") as excinfo:
            res.set_dims({"unknown_dim": 128})
        assert "unknown_dim" in str(excinfo)
