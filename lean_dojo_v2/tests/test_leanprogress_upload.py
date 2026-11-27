"""Tests for the LeanProgress Hugging Face release script."""

import json
import re
from pathlib import Path

import pytest

from lean_dojo_v2.lean_progress.create_sample_dataset import SAMPLE_DATA


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validate_sample_rows_and_statistics(tmp_path: Path) -> None:
    from lean_dojo_v2.lean_progress.upload_to_hf import validate_dataset

    dataset = tmp_path / "samples.jsonl"
    write_jsonl(dataset, SAMPLE_DATA)

    rows, stats, errors = validate_dataset(dataset)

    assert rows == SAMPLE_DATA
    assert stats == {
        "n_rows": 3,
        "steps_remaining": {"min": 0, "max": 12, "mean": 13 / 3},
        "rows_with_prefix": 1,
    }
    assert errors == []


def test_validation_skips_bad_rows_and_reports_line_numbers(tmp_path: Path) -> None:
    from lean_dojo_v2.lean_progress.upload_to_hf import validate_dataset

    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"goal":"g", "tactic":"t", "steps_remaining":-1}',
                '{"goal":"g", "steps_remaining":1}',
                '{"goal":"g", "tactic":"t", "steps_remaining":"1"}',
                '{"goal":"g", "tactic":"t", "steps_remaining":true}',
                "{broken json",
                '{"goal":"g", "tactic":"t", "steps_remaining":0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, stats, errors = validate_dataset(dataset)

    assert rows == [{"goal": "g", "tactic": "t", "steps_remaining": 0, "prefix": ""}]
    assert stats["n_rows"] == 1
    assert len(errors) == 5
    for line_number, error in enumerate(errors, 1):
        assert error.startswith(f"line {line_number}:")
    assert "steps_remaining" in errors[0]
    assert "tactic" in errors[1]
    assert "steps_remaining" in errors[2]
    assert "steps_remaining" in errors[3]
    assert "json" in errors[4].lower()


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ({"tactic": "t", "steps_remaining": 0}, "goal"),
        ({"goal": "", "tactic": "t", "steps_remaining": 0}, "goal"),
        ({"goal": 1, "tactic": "t", "steps_remaining": 0}, "goal"),
        ({"goal": "g", "tactic": "", "steps_remaining": 0}, "tactic"),
        ({"goal": "g", "tactic": 1, "steps_remaining": 0}, "tactic"),
        ({"goal": "g", "tactic": "t"}, "steps_remaining"),
        ({"goal": "g", "tactic": "t", "steps_remaining": 1.0}, "steps_remaining"),
        ({"goal": "g", "tactic": "t", "steps_remaining": 1, "prefix": None}, "prefix"),
        ([], "object"),
        (None, "object"),
    ],
)
def test_validation_rejects_invalid_fields_and_non_objects(
    tmp_path: Path, row: object, reason: str
) -> None:
    from lean_dojo_v2.lean_progress.upload_to_hf import validate_dataset

    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rows, stats, errors = validate_dataset(dataset)

    assert rows == []
    assert stats == {
        "n_rows": 0,
        "steps_remaining": {"min": 0, "max": 0, "mean": 0.0},
        "rows_with_prefix": 0,
    }
    assert len(errors) == 1
    assert errors[0].startswith("line 1:")
    assert reason in errors[0]


def test_validation_missing_file_raises(tmp_path: Path) -> None:
    from lean_dojo_v2.lean_progress.upload_to_hf import validate_dataset

    with pytest.raises(FileNotFoundError):
        validate_dataset(tmp_path / "missing.jsonl")


def test_dataset_card_describes_release_and_load_snippet() -> None:
    from lean_dojo_v2.lean_progress.upload_to_hf import build_dataset_card

    stats = {
        "n_rows": 3,
        "steps_remaining": {"min": 0, "max": 12, "mean": 13 / 3},
        "rows_with_prefix": 1,
    }
    card = build_dataset_card("example/proof-data", stats, "apache-2.0")

    assert card.startswith("---\n")
    assert "- lean\n" in card
    assert "- theorem-proving\n" in card
    assert "- lean4\n" in card
    assert "apache-2.0" in card
    assert "language: en" in card
    assert "generated for the release" in card.lower()
    assert "3" in card
    assert "0" in card and "12" in card and str(13 / 3) in card
    assert all(
        field in card for field in ("goal", "prefix", "tactic", "steps_remaining")
    )
    assert 'load_dataset("example/proof-data")' in card


def test_model_card_reports_model_and_metrics_without_inventing_values() -> None:
    from lean_dojo_v2.lean_progress.upload_to_hf import build_model_card

    card = build_model_card("example/model", "example/base-model")

    assert card.startswith("---\n")
    assert "- lean\n" in card and "- theorem-proving\n" in card
    assert "library_name: transformers" in card
    assert "license: mit" in card
    assert "example/model" in card and "example/base-model" in card
    assert "regression" in card and "steps_remaining" in card
    assert "training data" in card.lower()
    assert "metrics" in card.lower() and "after training" in card.lower()
    assert not re.search(r"\b(?:mse|mae|rmse)\s*[:=]\s*\d", card, re.I)


def test_dry_run_writes_card_and_manifest_without_touching_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    from lean_dojo_v2.lean_progress import upload_to_hf

    class ExplodingApi:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("network must not be touched in dry-run")

    stub = types.ModuleType("huggingface_hub")
    stub.HfApi = ExplodingApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    dataset = tmp_path / "samples.jsonl"
    write_jsonl(dataset, SAMPLE_DATA)
    out_dir = tmp_path / "out"

    manifest = upload_to_hf.upload_dataset(
        dataset, "example/proof-data", dry_run=True, out_dir=out_dir
    )

    assert manifest["dry_run"] is True
    assert manifest["kind"] == "dataset"
    assert manifest["n_rows"] == 3
    assert manifest["repo_id"] == "example/proof-data"
    card = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "example/proof-data" in card
    assert json.loads((out_dir / "manifest.json").read_text()) == manifest


def test_upload_without_token_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lean_dojo_v2.lean_progress import upload_to_hf

    monkeypatch.delenv("HF_TOKEN", raising=False)
    dataset = tmp_path / "samples.jsonl"
    write_jsonl(dataset, SAMPLE_DATA)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        upload_to_hf.upload_dataset(dataset, "example/proof-data")


class RecordingApi:
    """Stand-in for huggingface_hub.HfApi that records calls."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.calls: list[tuple] = []

    def create_repo(self, repo_id: str, repo_type: str, private: bool) -> None:
        self.calls.append(("create_repo", repo_id, repo_type, private))

    def upload_file(
        self, path_or_fileobj: object, path_in_repo: str, repo_id: str, repo_type: str
    ) -> None:
        self.calls.append(("upload_file", path_in_repo, repo_id, repo_type))

    def upload_folder(self, folder_path: str, repo_id: str, repo_type: str) -> None:
        self.calls.append(("upload_folder", folder_path, repo_id, repo_type))


def _install_recording_api(monkeypatch: pytest.MonkeyPatch) -> RecordingApi:
    import sys
    import types

    api = RecordingApi()
    stub = types.ModuleType("huggingface_hub")
    stub.HfApi = lambda token=None: api
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
    return api


def test_real_upload_records_expected_hub_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lean_dojo_v2.lean_progress import upload_to_hf

    api = _install_recording_api(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    dataset = tmp_path / "samples.jsonl"
    write_jsonl(dataset, SAMPLE_DATA)

    manifest = upload_to_hf.upload_dataset(dataset, "example/proof-data")

    assert manifest["dry_run"] is False
    assert ("create_repo", "example/proof-data", "dataset", False) in api.calls
    expected = ("upload_file", "samples.jsonl", "example/proof-data", "dataset")
    assert expected in api.calls
    assert ("upload_file", "README.md", "example/proof-data", "dataset") in api.calls


def test_model_upload_requires_config_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lean_dojo_v2.lean_progress import upload_to_hf

    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("fake", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="config.json"):
        upload_to_hf.upload_model(model_dir, "example/model", "example/base")


def test_model_dry_run_and_real_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lean_dojo_v2.lean_progress import upload_to_hf

    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_text("fake", encoding="utf-8")
    out_dir = tmp_path / "out"

    manifest = upload_to_hf.upload_model(
        model_dir, "example/model", "example/base", dry_run=True, out_dir=out_dir
    )

    assert manifest["kind"] == "model"
    assert manifest["base_model"] == "example/base"
    assert "config.json" in manifest["files"]
    assert "example/model" in (out_dir / "README.md").read_text(encoding="utf-8")

    api = _install_recording_api(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    upload_to_hf.upload_model(model_dir, "example/model", "example/base")
    assert ("create_repo", "example/model", "model", False) in api.calls
    assert ("upload_folder", str(model_dir), "example/model", "model") in api.calls


def test_cli_dry_run_and_invalid_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lean_dojo_v2.lean_progress import upload_to_hf

    dataset = tmp_path / "samples.jsonl"
    write_jsonl(dataset, SAMPLE_DATA)
    out_dir = tmp_path / "release"

    assert (
        upload_to_hf.main(
            [
                "--dataset",
                str(dataset),
                "--dry-run",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (out_dir / "dataset" / "README.md").is_file()
    assert (out_dir / "dataset" / "manifest.json").is_file()

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"goal":"g", "tactic":"t", "steps_remaining":-1}\n', encoding="utf-8"
    )
    assert upload_to_hf.main(["--dataset", str(bad), "--dry-run"]) == 2
    assert upload_to_hf.main(["--dataset", str(tmp_path / "missing.jsonl")]) == 2
