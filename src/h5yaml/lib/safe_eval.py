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
"""Evaluate simple mathematical expressions securely."""

from __future__ import annotations

import ast
import operator


def safe_eval(expr: str) -> None:
    """Evaluate simple mathematical expressions securely."""
    # define allowed operator
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def eval_node(node: ast.node) -> int | float:
        """Perform the actual evaluation, using module `ast`."""
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            return allowed_operators[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            return allowed_operators[type(node.op)](operand)

        if isinstance(node, ast.Constant):
            return node.value

        raise TypeError("Unsupported expression")

    parsed = ast.parse(expr, mode="eval")
    return eval_node(parsed.body)
