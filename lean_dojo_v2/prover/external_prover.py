"""DeepSeek agent for Lean theorem proving."""

from typing import Optional

try:
    import sys as _sys
    if _sys.platform == "win32":
        raise ImportError("Windows: using lean_interact_backend")
    from pantograph.expr import GoalState, Tactic
except ImportError:
    from lean_dojo_v2.interaction.lean_interact_backend import GoalState, Tactic  # type: ignore[assignment]

from lean_dojo_v2.database.models.theorems import Theorem
from lean_dojo_v2.external_api.python.external_models import HFTacticGenerator
from lean_dojo_v2.prover.base_prover import BaseProver


class ExternalProver(BaseProver):
    """DeepSeek-based agent for Lean theorem proving.

    This agent uses the DeepSeek-Prover-V2 model to generate tactics
    for theorem proving without retrieval augmentation.
    """

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-Prover-V2-671B:novita"):
        super().__init__()
        self.tactic_generator = HFTacticGenerator(model_name=model_name)

    def next_tactic(
        self,
        state: GoalState,
        goal_id: int,
    ) -> Optional[Tactic]:
        """Generate the next tactic using DeepSeek model."""
        return self.tactic_generator.generate(str(state))

    def generate_whole_proof(self, theorem: Theorem) -> str:
        self.theorem = theorem
        return self.tactic_generator.generate_whole_proof(str(self.theorem))
