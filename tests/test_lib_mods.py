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
"""Test module for h5yaml function `safe_eval`."""

from __future__ import annotations

import numpy as np
import pytest

from h5yaml.lib.adjust_attr import adjust_attr
from h5yaml.lib.safe_eval import safe_eval


class TestSafeEval:
    """Class to test H5Yaml from h5yaml.safe_eval."""

    def test_exceptions(self: TestSafeEval) -> None:
        """Unit-test for exeptions."""
        with pytest.raises(KeyError, match=r"ast.Mod") as excinfo:
            safe_eval("2 % 3")
        assert "<class 'ast.Mod'>" in str(excinfo)
        with pytest.raises(KeyError, match=r"ast.FloorDiv") as excinfo:
            safe_eval("2 // 3")
        assert "<class 'ast.FloorDiv'>" in str(excinfo)
        with pytest.raises(KeyError, match="Unsupported expression") as excinfo:
            safe_eval("log(3)")
        assert "Unsupported expression" in str(excinfo)

    def test_unary(self: TestSafeEval) -> None:
        """Unit-test for unary operations."""
        assert safe_eval("not False")
        assert safe_eval("~0x1") == -2

    def test_binary(self: TestSafeEval) -> None:
        """Unit-test for binary operations."""
        assert safe_eval("0x123") == 291
        assert safe_eval("2 + 3") == 5
        assert safe_eval("2 - 3") == -1
        assert safe_eval("2 * 3") == 6
        assert safe_eval("3 / 2") == 1.5


class TestAdjustAttr:
    """Class to test H5Yaml from h5yaml.adjust_attr."""

    def test_flag(self: TestSafeEval) -> None:
        """Unit-test for flag_values and flag_mask attributes."""
        attr_val = [0, 1, 2, 4, 8]
        assert np.array_equiv(
            adjust_attr("u1", "flag_values", attr_val), np.array(attr_val, dtype="u1")
        )
        attr_val = [0x7F, 0x3F80, 0x1FC000, 0xFE00000]
        assert np.array_equiv(
            adjust_attr("u4", "flag_masks", attr_val), np.array(attr_val, dtype="u4")
        )

    def test_scale(self: TestSafeEval) -> None:
        """Unit-test for scale_factor attributes."""
        attr_val = "2 * 3.14"
        assert adjust_attr("u1", "scale_factor", attr_val) == safe_eval(attr_val)

    def test_valid(self: TestSafeEval) -> None:
        """Unit-test for valid_* attributes."""
        attr_val = "123"
        assert isinstance(adjust_attr("i1", "valid_min", attr_val), np.int8)
        assert isinstance(adjust_attr("i2", "valid_max", attr_val), np.int16)
        assert isinstance(adjust_attr("i4", "valid_min", attr_val), np.int32)
        assert isinstance(adjust_attr("i8", "valid_max", attr_val), np.int64)
        assert isinstance(adjust_attr("u1", "valid_min", attr_val), np.uint8)
        assert isinstance(adjust_attr("u2", "valid_max", attr_val), np.uint16)
        assert isinstance(adjust_attr("u4", "valid_min", attr_val), np.uint32)
        assert isinstance(adjust_attr("u8", "valid_max", attr_val), np.uint64)
        assert isinstance(adjust_attr("f2", "valid_min", attr_val), np.float16)
        assert isinstance(adjust_attr("f4", "valid_max", attr_val), np.float32)
        assert isinstance(adjust_attr("f8", "valid_min", attr_val), np.float64)
        assert isinstance(adjust_attr("xx", "valid_max", attr_val), type(attr_val))

        attr_range = [2, 200]
        assert adjust_attr("f8", "valid_range", attr_range).dtype == np.dtype("float64")
        assert adjust_attr("xx", "valid_range", attr_range).dtype == np.dtype("int64")
