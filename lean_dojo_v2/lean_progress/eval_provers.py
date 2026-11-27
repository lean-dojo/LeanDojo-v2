"""Exercise prover search and report claims, verification, and faithfulness.

A claimed pass is a successful runner result with no tactic equal to sorry;
the harness rejects sorry statically because it is never a proof. A verified
pass is a claimed pass confirmed by a Lean server, and only verified passes
count in pass_rate.

For claimed passes, faithfulness is the mean absolute error between the
LeanProgress scorer's predicted steps_remaining for the first proof step and
the actual number of tactics in the returned proof, plus the fraction within
one of that number. Faithfulness is not measured without a scorer.
mean_steps_used is the mean number of search steps (iterations) over claimed
passes, which can exceed the number of tactics in the returned proof.

Known limits: goal-only search cannot produce tactics for HFProver or
RetrievalProver. Both next_tactic implementations return None unless the
prover holds a traced Theorem object, and goal-only search leaves it unset,
so those runners report no claimed passes on goal strings. Only
ExternalProver (and the fake runner) run end to end on goal strings.
Theorem-backed evaluation through the traced database is not implemented
in this PR.
"""

import argparse
import json
import sys


class FakeProverRunner:
    """Simulate successful searches without claiming Lean verification."""

    def run(self, theorem):
        tactics = list(theorem["expected_tactics"])
        return {
            "success": True,
            "tactics": tactics,
            "steps": len(tactics),
            "verified": False,
            "error": None,
        }


class _RealProverRunner:
    @staticmethod
    def _check_max_steps(max_steps):
        if max_steps != 100:
            raise RuntimeError(
                "BaseProver.search only supports its fixed limit of 100 steps"
            )

    @staticmethod
    def _make_server(server_class):
        try:
            return server_class()
        except Exception as exc:
            raise RuntimeError(f"Pantograph Server creation failed: {exc}") from exc

    def run(self, theorem):
        try:
            result, used_tactics = self.prover.search(self.server, goal=theorem["goal"])
            return {
                "success": bool(result.success),
                "tactics": (
                    [str(tactic) for tactic in used_tactics] if used_tactics else []
                ),
                "steps": int(result.steps),
                "verified": bool(result.success),
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "tactics": [],
                "steps": 0,
                "verified": False,
                "error": str(exc),
            }


class HFProverRunner(_RealProverRunner):
    """Run HFProver search on a goal string.

    Known limit: HFProver.next_tactic returns None unless the prover holds
    a traced Theorem object, and goal-only search leaves it unset, so this
    runner produces no tactics on goal strings. Theorem-backed evaluation
    is not implemented in this PR.
    """

    def __init__(self, model: str | None = None, max_steps: int = 100):
        self._check_max_steps(max_steps)
        try:
            from pantograph import Server
        except Exception as exc:
            raise RuntimeError(f"Pantograph Server import failed: {exc}") from exc
        try:
            from lean_dojo_v2.prover.hf_prover import HFProver
        except Exception as exc:
            raise RuntimeError(f"HFProver import failed: {exc}") from exc
        if model is None:
            raise RuntimeError("HFProver requires a model checkpoint")
        self.server = self._make_server(Server)
        try:
            self.prover = HFProver(ckpt_path=model)
        except Exception as exc:
            raise RuntimeError(f"HFProver initialization failed: {exc}") from exc


class RetrievalProverRunner(_RealProverRunner):
    """Report the missing retrieval configuration in this harness interface.

    RetrievalProver needs three checkpoint paths, and like HFProver its
    next_tactic returns None without a traced Theorem object, so goal-only
    evaluation cannot run it even with the paths supplied.
    """

    def __init__(self, model: str | None = None, max_steps: int = 100):
        self._check_max_steps(max_steps)
        try:
            from pantograph import Server
        except Exception as exc:
            raise RuntimeError(f"Pantograph Server import failed: {exc}") from exc
        try:
            from lean_dojo_v2.prover.retrieval_prover import RetrievalProver
        except Exception as exc:
            raise RuntimeError(f"RetrievalProver import failed: {exc}") from exc
        raise RuntimeError(
            "RetrievalProver requires ret_ckpt_path, gen_ckpt_path, and "
            "indexed_corpus_path; a single model string cannot supply them. "
            "It also needs a traced Theorem object: its next_tactic returns "
            "None on goal-only search."
        )


class ExternalProverRunner(_RealProverRunner):
    def __init__(self, model: str | None = None, max_steps: int = 100):
        self._check_max_steps(max_steps)
        try:
            from pantograph import Server
        except Exception as exc:
            raise RuntimeError(f"Pantograph Server import failed: {exc}") from exc
        try:
            from lean_dojo_v2.prover.external_prover import ExternalProver
        except Exception as exc:
            raise RuntimeError(f"ExternalProver import failed: {exc}") from exc
        self.server = self._make_server(Server)
        try:
            self.prover = (
                ExternalProver(model_name=model)
                if model is not None
                else ExternalProver()
            )
        except Exception as exc:
            raise RuntimeError(f"ExternalProver initialization failed: {exc}") from exc


def evaluate(theorems, runner, scorer=None, max_steps=100):
    """Evaluate results; any search limit must be configured on the runner."""
    per_theorem = []
    claimed_goals = []
    for theorem in theorems:
        result = runner.run(theorem)
        tactics = result["tactics"]
        claimed = bool(result["success"]) and not any(
            tactic.strip().lower() == "sorry" for tactic in tactics
        )
        per_theorem.append(
            {
                "id": theorem["id"],
                "claimed": claimed,
                "verified": claimed and bool(result.get("verified", False)),
                "steps": result["steps"],
                "tactics": tactics,
                "error": result.get("error"),
            }
        )
        if claimed:
            claimed_goals.append(theorem["goal"])

    claimed_results = [entry for entry in per_theorem if entry["claimed"]]
    n_theorems = len(per_theorem)
    verified_passes = sum(entry["verified"] for entry in per_theorem)
    if scorer is None:
        faithfulness = {"measured": False, "reason": "No scorer provided"}
    elif not claimed_results:
        faithfulness = {"measured": False, "reason": "No claimed passes"}
    elif any(not entry["tactics"] for entry in claimed_results):
        faithfulness = {
            "measured": False,
            "reason": "A claimed pass has no first tactic to score",
        }
    else:
        errors = [
            abs(
                float(scorer.predict(goal, prefix="", tactic=entry["tactics"][0]))
                - len(entry["tactics"])
            )
            for goal, entry in zip(claimed_goals, claimed_results)
        ]
        faithfulness = {
            "measured": True,
            "mae": sum(errors) / len(errors),
            "within_one": sum(error <= 1 for error in errors) / len(errors),
            "n_measured": len(errors),
        }
    return {
        "n_theorems": n_theorems,
        "claimed_passes": len(claimed_results),
        "verified_passes": verified_passes,
        "pass_rate": verified_passes / n_theorems if n_theorems else 0.0,
        "mean_steps_used": (
            sum(entry["steps"] for entry in claimed_results) / len(claimed_results)
            if claimed_results
            else 0.0
        ),
        "faithfulness": faithfulness,
        "per_theorem": per_theorem,
    }


def main(argv=None):
    from lean_dojo_v2.lean_progress.theorems import load_theorems

    parser = argparse.ArgumentParser(
        description="Run a prover on the smoke theorem set"
    )
    parser.add_argument("--theorems", help="JSONL theorem file")
    parser.add_argument(
        "--prover", choices=("fake", "hf", "retrieval", "external"), default="fake"
    )
    parser.add_argument("--model")
    parser.add_argument("--output", help="JSON report path")
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args(argv)

    try:
        theorems = load_theorems(args.theorems)
    except (OSError, ValueError) as exc:
        print(f"Failed to load theorems: {exc}", file=sys.stderr)
        return 2

    runners = {
        "fake": FakeProverRunner,
        "hf": HFProverRunner,
        "retrieval": RetrievalProverRunner,
        "external": ExternalProverRunner,
    }
    try:
        runner = (
            FakeProverRunner()
            if args.prover == "fake"
            else runners[args.prover](model=args.model, max_steps=args.max_steps)
        )
    except RuntimeError as exc:
        print(f"Failed to construct {args.prover} prover: {exc}", file=sys.stderr)
        return 2

    report = evaluate(theorems, runner, max_steps=args.max_steps)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as destination:
                json.dump(report, destination, ensure_ascii=False, indent=2)
                destination.write("\n")
        except OSError as exc:
            print(f"Failed to write report: {exc}", file=sys.stderr)
            return 2

    print(f"Theorems: {report['n_theorems']}")
    print(f"Claimed passes: {report['claimed_passes']}")
    print(f"Verified passes: {report['verified_passes']}")
    print(f"Pass rate: {report['pass_rate']:.2%}")
    faithfulness = report["faithfulness"]
    if faithfulness["measured"]:
        print(
            f"Faithfulness: MAE {faithfulness['mae']:.3f}, "
            f"within one {faithfulness['within_one']:.2%}"
        )
    else:
        print(f"Faithfulness: not measured ({faithfulness['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
