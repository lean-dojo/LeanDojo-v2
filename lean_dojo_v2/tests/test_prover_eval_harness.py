import json
import subprocess
import sys
import types

import pytest


def test_bundled_smoke_set_contains_requested_theorems():
    from lean_dojo_v2.lean_progress.theorems import load_theorems

    theorems = load_theorems()

    assert theorems == [
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
    theorems[0]["expected_tactics"].append("changed")
    assert load_theorems()[0]["expected_tactics"] == [
        "intro p q h",
        "exact ⟨h.2, h.1⟩",
    ]


@pytest.mark.parametrize("missing_key", ["id", "goal", "expected_tactics"])
def test_jsonl_rejects_missing_required_fields(tmp_path, missing_key):
    from lean_dojo_v2.lean_progress.theorems import load_theorems

    row = {"id": "example", "goal": "True", "expected_tactics": ["trivial"]}
    del row[missing_key]
    path = tmp_path / "theorems.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=missing_key):
        load_theorems(path)


def test_jsonl_loads_all_rows(tmp_path):
    from lean_dojo_v2.lean_progress.theorems import load_theorems

    path = tmp_path / "theorems.jsonl"
    path.write_text(
        '{"id": "a", "goal": "True", "expected_tactics": ["trivial"]}\n'
        '{"id": "b", "goal": "False", "expected_tactics": []}\n',
        encoding="utf-8",
    )

    assert load_theorems(path) == [
        {"id": "a", "goal": "True", "expected_tactics": ["trivial"]},
        {"id": "b", "goal": "False", "expected_tactics": []},
    ]


def test_fake_runner_never_verifies_and_rejects_sorry():
    from lean_dojo_v2.lean_progress.eval_provers import FakeProverRunner, evaluate
    from lean_dojo_v2.lean_progress.theorems import load_theorems

    theorems = load_theorems()
    runner = FakeProverRunner()
    outcome = runner.run(theorems[0])
    assert outcome == {
        "success": True,
        "tactics": ["intro p q h", "exact ⟨h.2, h.1⟩"],
        "steps": 2,
        "verified": False,
        "error": None,
    }
    outcome["tactics"].append("changed")
    assert len(theorems[0]["expected_tactics"]) == 2

    report = evaluate(theorems, runner)
    assert report["n_theorems"] == 4
    assert report["claimed_passes"] == 3
    assert report["verified_passes"] == 0
    assert report["pass_rate"] == 0.0
    assert report["mean_steps_used"] == 2.0
    assert report["faithfulness"]["measured"] is False
    assert isinstance(report["faithfulness"]["reason"], str)
    assert report["per_theorem"][-1]["id"] == "group_comm_sorry"
    assert report["per_theorem"][-1]["claimed"] is False


def test_success_alone_does_not_verify_and_sorry_is_case_insensitive():
    from lean_dojo_v2.lean_progress.eval_provers import evaluate
    from lean_dojo_v2.lean_progress.theorems import load_theorems

    class StubRunner:
        def run(self, theorem):
            return {
                "success": True,
                "tactics": [
                    "  SoRrY  " if tactic == "sorry" else tactic
                    for tactic in theorem["expected_tactics"]
                ],
                "steps": len(theorem["expected_tactics"]),
                "error": None,
            }

    report = evaluate(load_theorems(), StubRunner())
    assert report["claimed_passes"] == 3
    assert report["verified_passes"] == 0
    assert report["per_theorem"][-1]["claimed"] is False


def test_empty_theorem_set_has_no_passes_or_faithfulness():
    from lean_dojo_v2.lean_progress.eval_provers import FakeProverRunner, evaluate

    class Scorer:
        def predict(self, goal, prefix, tactic):
            raise AssertionError("There is no theorem to score")

    report = evaluate([], FakeProverRunner(), scorer=Scorer())
    assert report["n_theorems"] == 0
    assert report["pass_rate"] == 0.0
    assert report["mean_steps_used"] == 0.0
    assert report["faithfulness"]["measured"] is False
    assert report["per_theorem"] == []


def test_faithfulness_scores_first_tactic_against_proof_length():
    from lean_dojo_v2.lean_progress.eval_provers import FakeProverRunner, evaluate

    theorems = [
        {"id": "a", "goal": "goal a", "expected_tactics": ["one"]},
        {"id": "b", "goal": "goal b", "expected_tactics": ["one", "two"]},
        {
            "id": "c",
            "goal": "goal c",
            "expected_tactics": ["one", "two", "three", "four"],
        },
        {"id": "control", "goal": "false", "expected_tactics": ["sorry"]},
    ]

    class Scorer:
        def __init__(self):
            self.calls = []

        def predict(self, goal, prefix, tactic):
            self.calls.append((goal, prefix, tactic))
            return 2.0

    class SearchStepsRunner(FakeProverRunner):
        def run(self, theorem):
            result = super().run(theorem)
            result["steps"] = 10
            return result

    scorer = Scorer()
    report = evaluate(theorems, SearchStepsRunner(), scorer=scorer)
    assert report["mean_steps_used"] == 10.0
    assert report["faithfulness"] == {
        "measured": True,
        "mae": 1.0,
        "within_one": 2 / 3,
        "n_measured": 3,
    }
    assert scorer.calls == [
        ("goal a", "", "one"),
        ("goal b", "", "one"),
        ("goal c", "", "one"),
    ]


def test_scorer_cannot_measure_a_claim_with_no_first_tactic():
    from lean_dojo_v2.lean_progress.eval_provers import FakeProverRunner, evaluate

    class Scorer:
        def predict(self, goal, prefix, tactic):
            raise AssertionError("There is no first tactic to score")

    report = evaluate(
        [{"id": "empty", "goal": "True", "expected_tactics": []}],
        FakeProverRunner(),
        scorer=Scorer(),
    )
    assert report["claimed_passes"] == 1
    assert report["faithfulness"]["measured"] is False
    assert "tactic" in report["faithfulness"]["reason"].lower()


def test_verified_passes_require_both_a_claim_and_verification():
    from lean_dojo_v2.lean_progress.eval_provers import evaluate

    class Runner:
        def run(self, theorem):
            return {
                "success": theorem["id"] != "failed",
                "tactics": theorem["expected_tactics"],
                "steps": 1,
                "verified": True,
                "error": None,
            }

    theorems = [
        {"id": "verified", "goal": "True", "expected_tactics": ["trivial"]},
        {"id": "sorry", "goal": "False", "expected_tactics": ["sorry"]},
        {"id": "failed", "goal": "True", "expected_tactics": ["trivial"]},
    ]
    report = evaluate(theorems, Runner())
    assert report["claimed_passes"] == 1
    assert report["verified_passes"] == 1
    assert report["pass_rate"] == 1 / 3
    assert [entry["verified"] for entry in report["per_theorem"]] == [
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    ("runner_name", "module_name", "prover_name", "constructor_args"),
    [
        ("HFProverRunner", "hf_prover", "HFProver", {"ckpt_path": "checkpoint"}),
        (
            "ExternalProverRunner",
            "external_prover",
            "ExternalProver",
            {"model_name": "checkpoint"},
        ),
    ],
)
def test_real_runner_search_maps_result_and_errors(
    monkeypatch, runner_name, module_name, prover_name, constructor_args
):
    from lean_dojo_v2.lean_progress import eval_provers

    class Server:
        pass

    class Prover:
        def __init__(self, **kwargs):
            assert kwargs == constructor_args

        def search(self, server, goal):
            assert isinstance(server, Server)
            if goal == "bad":
                raise RuntimeError("search failed")
            if goal == "False":
                return types.SimpleNamespace(success=False, steps=5), None
            assert goal == "True"
            return types.SimpleNamespace(success=True, steps=4), [Tactic()]

    class Tactic:
        def __str__(self):
            return "trivial"

    fake_server_module = types.ModuleType("pantograph")
    fake_server_module.Server = Server
    fake_prover_package = types.ModuleType("lean_dojo_v2.prover")
    fake_prover_package.__path__ = []
    fake_prover_module = types.ModuleType(f"lean_dojo_v2.prover.{module_name}")
    setattr(fake_prover_module, prover_name, Prover)
    monkeypatch.setitem(sys.modules, "pantograph", fake_server_module)
    monkeypatch.setitem(sys.modules, "lean_dojo_v2.prover", fake_prover_package)
    monkeypatch.setitem(
        sys.modules, f"lean_dojo_v2.prover.{module_name}", fake_prover_module
    )

    runner = getattr(eval_provers, runner_name)(model="checkpoint")
    assert runner.run({"goal": "True"}) == {
        "success": True,
        "tactics": ["trivial"],
        "steps": 4,
        "verified": True,
        "error": None,
    }
    assert runner.run({"goal": "bad"}) == {
        "success": False,
        "tactics": [],
        "steps": 0,
        "verified": False,
        "error": "search failed",
    }
    assert runner.run({"goal": "False"}) == {
        "success": False,
        "tactics": [],
        "steps": 5,
        "verified": False,
        "error": None,
    }

    class BrokenServer:
        def __init__(self):
            raise OSError("lean executable absent")

    fake_server_module.Server = BrokenServer
    with pytest.raises(RuntimeError, match="Pantograph Server.*lean executable absent"):
        getattr(eval_provers, runner_name)(model="checkpoint")


def test_real_runner_reports_missing_pantograph(monkeypatch):
    from lean_dojo_v2.lean_progress.eval_provers import HFProverRunner

    monkeypatch.setitem(sys.modules, "pantograph", None)
    with pytest.raises(RuntimeError, match="[Pp]antograph"):
        HFProverRunner(model=None)


def test_real_runner_rejects_step_limits_unsupported_by_search():
    from lean_dojo_v2.lean_progress.eval_provers import ExternalProverRunner

    with pytest.raises(RuntimeError, match="100"):
        ExternalProverRunner(model=None, max_steps=10)


def test_retrieval_runner_reports_required_paths(monkeypatch):
    from lean_dojo_v2.lean_progress.eval_provers import RetrievalProverRunner

    fake_server_module = types.ModuleType("pantograph")
    fake_server_module.Server = object
    fake_prover_package = types.ModuleType("lean_dojo_v2.prover")
    fake_prover_package.__path__ = []
    fake_prover_module = types.ModuleType("lean_dojo_v2.prover.retrieval_prover")
    fake_prover_module.RetrievalProver = object
    monkeypatch.setitem(sys.modules, "pantograph", fake_server_module)
    monkeypatch.setitem(sys.modules, "lean_dojo_v2.prover", fake_prover_package)
    monkeypatch.setitem(
        sys.modules, "lean_dojo_v2.prover.retrieval_prover", fake_prover_module
    )

    with pytest.raises(
        RuntimeError, match="ret_ckpt_path.*gen_ckpt_path.*indexed_corpus_path"
    ):
        RetrievalProverRunner(model="checkpoint")


def test_fake_cli_writes_json_report(tmp_path):
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lean_dojo_v2.lean_progress.eval_provers",
            "--prover",
            "fake",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report) == {
        "n_theorems",
        "claimed_passes",
        "verified_passes",
        "pass_rate",
        "mean_steps_used",
        "faithfulness",
        "per_theorem",
    }
    assert (
        report["n_theorems"],
        report["claimed_passes"],
        report["verified_passes"],
    ) == (
        4,
        3,
        0,
    )
    assert "faithfulness" in result.stdout.lower()


def test_hf_cli_reports_missing_pantograph(monkeypatch, capsys):
    from lean_dojo_v2.lean_progress.eval_provers import main

    monkeypatch.setitem(sys.modules, "pantograph", None)
    assert main(["--prover", "hf"]) == 2
    assert "pantograph" in capsys.readouterr().err.lower()


def test_cli_reports_theorem_loading_errors(tmp_path):
    source = tmp_path / "broken.jsonl"
    source.write_text('{"id": "bad", "expected_tactics": []}\n', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lean_dojo_v2.lean_progress.eval_provers",
            "--theorems",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "goal" in result.stderr.lower()
