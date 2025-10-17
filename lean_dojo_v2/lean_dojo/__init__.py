import os

from loguru import logger

from lean_dojo_v2.utils.constants import __version__

from .data_extraction.dataset import generate_benchmark
from .data_extraction.lean import LeanFile, LeanGitRepo, Pos, Theorem, get_latest_commit
from .data_extraction.trace import get_traced_repo_path, is_available_in_cache, trace
from .data_extraction.traced_data import (
    TracedFile,
    TracedRepo,
    TracedTactic,
    TracedTheorem,
)
from .interaction.dojo import (
    CommandState,
    Dojo,
    DojoCrashError,
    DojoInitError,
    DojoTacticTimeoutError,
    LeanError,
    ProofFinished,
    ProofGivenUp,
    TacticResult,
    TacticState,
    check_proof,
)
from .interaction.parse_goals import Declaration, Goal, parse_goals
