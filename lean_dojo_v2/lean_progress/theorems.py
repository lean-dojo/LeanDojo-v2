"""A small fixed SMOKE set for exercising the harness, not a benchmark.

No entry has been verified against a Lean toolchain in this PR; only a Lean
server verdict counts as a pass.
"""

import json
from copy import deepcopy

THEOREMS = [
    {
        "id": "and_comm",
        "goal": "∀ {p q : Prop}, p ∧ q → q ∧ p",
        "expected_tactics": ["intro p q h", "exact ⟨h.2, h.1⟩"],
        "provable": True,
        "note": "README goal example",
    },
    {
        "id": "gcd_self",
        "goal": "∀ n : ℕ, Nat.gcd n n = n",
        "expected_tactics": ["intro n", "exact Nat.gcd_self n"],
        "provable": True,
        "note": "uses Nat.gcd_self, as in the repo sample dataset",
    },
    {
        "id": "eq_symm",
        "goal": "∀ {a b : ℝ}, a = b → b = a",
        "expected_tactics": ["intro a b h", "exact h.symm"],
        "provable": True,
        "note": "uses h.symm, as in the repo sample dataset",
    },
    {
        "id": "group_comm_sorry",
        "goal": "∀ {G : Type} [Group G] (g h : G), g * h = h * g",
        "expected_tactics": ["sorry"],
        "provable": False,
        "note": (
            "control: false in general for non-abelian groups, "
            "and sorry is never a proof"
        ),
    },
]


def load_theorems(path=None):
    if path is None:
        return deepcopy(THEOREMS)

    rows = []
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} must be a JSON object")
            missing = {"id", "goal", "expected_tactics"} - row.keys()
            if missing:
                raise ValueError(
                    f"Line {line_number} missing required fields: "
                    f"{', '.join(sorted(missing))}"
                )
            rows.append(row)
    return rows
