"""
Windows-compatible Lean backend using lean-interact.

Drop-in replacement for PyPantograph on Windows/Linux/macOS.
Exports the same API surface as pantograph so existing code
(base_prover.py, hf_prover.py, external_prover.py) works unchanged.

PyPantograph API replicated:
  - Server              → LeanInteractServer
  - GoalState           → LIGoalState
  - Site                → Site (passthrough)
  - Tactic              → str (Lean 4 tactic string)
  - Agent               → Agent (ABC)
  - SearchResult        → SearchResult
  - SearchState         → SearchState
  - ServerError         → ServerError
  - TacticFailure       → TacticFailure

Usage:
    # Instead of:
    from pantograph import Server
    from pantograph.expr import GoalState, Site, Tactic
    from pantograph.search import Agent, SearchResult, SearchState
    from pantograph.server import ServerError, TacticFailure

    # Use:
    from lean_dojo_v2.interaction.lean_interact_backend import (
        Server, GoalState, Site, Tactic,
        Agent, SearchResult, SearchState,
        ServerError, TacticFailure,
    )
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from lean_interact import (
    AutoLeanServer, Command, LocalProject, LeanREPLConfig, ProofStep,
)
from lean_interact.interface import (
    CommandResponse, LeanError, ProofStepResponse,
)

# ──────────────────────────────────────────────────────────────────
# Type alias — Lean 4 tactic is just a string
# ──────────────────────────────────────────────────────────────────

Tactic = str


# ──────────────────────────────────────────────────────────────────
# Exceptions (pantograph API compatible)
# ──────────────────────────────────────────────────────────────────

class ServerError(Exception):
    pass


class TacticFailure(Exception):
    pass


# ──────────────────────────────────────────────────────────────────
# Goal  (pantograph.expr.Goal equivalent)
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Goal:
    pp: str          # pretty-printed goal string
    variables: list  = field(default_factory=list)

    def __str__(self) -> str:
        return self.pp


# ──────────────────────────────────────────────────────────────────
# GoalState  (pantograph.expr.GoalState equivalent)
# ──────────────────────────────────────────────────────────────────

@dataclass
class GoalState:
    state_id: int
    goals: list[Goal]

    @property
    def is_solved(self) -> bool:
        return len(self.goals) == 0

    def __str__(self) -> str:
        return "\n".join(g.pp for g in self.goals)

    @classmethod
    def _from_strings(cls, state_id: int, goal_strings: list[str]) -> "GoalState":
        goals = [Goal(pp=g) for g in goal_strings]
        return cls(state_id=state_id, goals=goals)


# ──────────────────────────────────────────────────────────────────
# Site  (pantograph.expr.Site equivalent)
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Site:
    goal_id: int = 0
    auto_resume: bool = True


# ──────────────────────────────────────────────────────────────────
# Server  (pantograph.Server equivalent)
# ──────────────────────────────────────────────────────────────────

class Server:
    """
    LeanInteract-backed Lean server.
    Replaces pantograph.Server with Windows-compatible implementation.
    """

    def __init__(
        self,
        project_path: str = ".",
        imports: Optional[list[str]] = None,
    ):
        project = LocalProject(directory=project_path)
        config  = LeanREPLConfig(project=project)
        self._server = AutoLeanServer(config)

        # Load any extra imports (e.g. Mathlib)
        if imports:
            import_str = "\n".join(f"import {i}" for i in imports)
            self._server.run(Command(cmd=import_str))

    def is_automatic(self) -> bool:
        """Always true — LeanInteract handles everything automatically."""
        return True

    def goal_start(self, goal: str) -> GoalState:
        """
        Open a proof for the given goal string.
        Equivalent to pantograph Server.goal_start().

        We wrap the goal in a synthetic theorem with `sorry`,
        then extract the proof state from the sorry.
        """
        synthetic = f"theorem __lean_evolve_goal__ : ({goal}) := by\n  sorry"
        r = self._server.run(Command(cmd=synthetic))

        if isinstance(r, LeanError):
            raise ServerError(f"Could not start goal: {r.message}")

        if r.sorries:
            s = r.sorries[0]
            return GoalState._from_strings(
                state_id=s.proof_state,
                goal_strings=[s.goal],
            )

        # No sorry means it was trivially proved or failed
        if any(m.severity == "error" for m in r.messages):
            errs = [m.data for m in r.messages if m.severity == "error"]
            raise ServerError(f"Goal start failed: {errs[0][:200]}")

        # Trivially solved (no goals)
        return GoalState(state_id=-1, goals=[])

    def goal_tactic(
        self,
        goal_state: GoalState,
        tactic: Tactic,
        site: Optional[Site] = None,
    ) -> GoalState:
        """
        Apply a tactic to a goal state.
        Equivalent to pantograph Server.goal_tactic().
        Raises TacticFailure on error.
        """
        ps = self._server.run(
            ProofStep(proofState=goal_state.state_id, tactic=tactic)
        )

        if isinstance(ps, LeanError):
            raise TacticFailure(ps.message)

        errors = [m for m in ps.messages if m.severity == "error"]
        if errors:
            raise TacticFailure(errors[0].data[:200])

        return GoalState._from_strings(
            state_id=ps.proof_state,
            goal_strings=ps.goals,
        )

    def env_inspect(self, full_name: str) -> dict:
        """
        Inspect a theorem/declaration type.
        Equivalent to pantograph Server.env_inspect().
        Returns {"type": {"pp": "<type string>"}}
        """
        code = f"#check @{full_name}"
        r = self._server.run(Command(cmd=code))

        # Extract from info messages
        for msg in getattr(r, "messages", []):
            if msg.severity == "info":
                return {"type": {"pp": msg.data.strip()}}

        return {"type": {"pp": full_name}}


# ──────────────────────────────────────────────────────────────────
# SearchState  (pantograph.search.SearchState equivalent)
# ──────────────────────────────────────────────────────────────────

@dataclass
class SearchState:
    goal_state: GoalState
    parent: Optional["SearchState"]
    parent_goal_id: Optional[int]
    priorities: list[float]
    trials: dict = field(default_factory=dict)
    tactic_feedback: Optional[str] = None

    @property
    def is_solved(self) -> bool:
        return self.goal_state.is_solved

    @property
    def next_goal_id(self) -> int:
        """Return goal with highest priority (most likely to be solvable)."""
        if not self.goal_state.goals:
            return 0
        if self.priorities:
            return max(
                range(len(self.goal_state.goals)),
                key=lambda i: self.priorities[i] if i < len(self.priorities) else 0.0,
            )
        return 0


# ──────────────────────────────────────────────────────────────────
# SearchResult  (pantograph.search.SearchResult equivalent)
# ──────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    n_goals_root: int
    duration: float
    success: bool
    steps: int


# ──────────────────────────────────────────────────────────────────
# Agent  (pantograph.search.Agent equivalent)
# ──────────────────────────────────────────────────────────────────

class Agent(ABC):
    """
    Abstract base for theorem-proving agents.
    Matches pantograph.search.Agent interface.
    """

    def reset(self):
        """Reset internal state between searches."""
        pass

    @abstractmethod
    def next_tactic(
        self,
        state: GoalState,
        goal_id: int,
    ) -> Optional[Tactic]:
        pass


# ──────────────────────────────────────────────────────────────────
# Platform check utility
# ──────────────────────────────────────────────────────────────────

def is_windows() -> bool:
    return sys.platform == "win32"


def get_backend():
    """
    Auto-select backend based on platform.
    Returns the module to use for Lean interaction.
    """
    if is_windows():
        return sys.modules[__name__]
    try:
        import pantograph
        return pantograph
    except ImportError:
        return sys.modules[__name__]
