"""
Solver backend helpers.

Centralizes backend selection so benchmark runs can prefer Gurobi when
available while retaining a clean CBC fallback.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from typing import Literal

import pulp


SolverBackend = Literal["auto", "gurobi", "cbc"]


@dataclass(frozen=True)
class SolverSelection:
    requested: SolverBackend
    active: str
    reason: str


def gurobi_available() -> bool:
    """Return whether gurobipy is importable in the current environment."""
    return importlib.util.find_spec("gurobipy") is not None


def resolve_backend(requested: SolverBackend = "auto") -> SolverSelection:
    """Resolve a requested backend to an actually usable backend."""
    if requested == "gurobi":
        if gurobi_available():
            return SolverSelection(requested=requested, active="gurobi", reason="gurobipy available")
        return SolverSelection(
            requested=requested,
            active="cbc",
            reason="gurobipy unavailable, falling back to CBC",
        )

    if requested == "cbc":
        return SolverSelection(requested=requested, active="cbc", reason="CBC requested")

    if gurobi_available():
        return SolverSelection(requested=requested, active="gurobi", reason="auto-selected Gurobi")
    return SolverSelection(requested=requested, active="cbc", reason="auto-selected CBC")


def build_pulp_solver(
    requested: SolverBackend = "auto",
    time_limit: int = 300,
    msg: bool = False,
):
    """Construct a PuLP solver instance and return it with backend metadata."""
    selection = resolve_backend(requested)

    if selection.active == "gurobi":
        solver = pulp.GUROBI(msg=msg, timeLimit=time_limit)
    else:
        cbc_path = shutil.which("cbc")
        if cbc_path:
            solver = pulp.COIN_CMD(path=cbc_path, msg=msg, timeLimit=time_limit)
        else:
            solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)

    return solver, selection
