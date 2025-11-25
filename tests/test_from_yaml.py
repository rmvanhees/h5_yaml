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
"""Test module for h5yaml function `conf_from_yaml`."""

from __future__ import annotations

from importlib.resources import files

import pytest

from h5yaml.conf_from_yaml import conf_from_yaml


def test_from_yaml() -> None:
    """..."""
    with pytest.raises(FileNotFoundError, match="not found") as excinfo:
        _ = conf_from_yaml(files("h5yaml.Data") / "not_existing.yaml")
    assert f"{files('h5yaml.Data') / 'not_existing.yaml'} not found" in str(excinfo)

    with pytest.raises(RuntimeError, match=r"Failed to parse .*") as excinfo:
        _ = conf_from_yaml("README.md")
    assert "Failed to parse" in str(excinfo)

    res = conf_from_yaml(files("h5yaml.Data") / "nc_testing.yaml")
    assert isinstance(res, dict)
