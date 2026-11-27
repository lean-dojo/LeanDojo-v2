"""Prepare and upload LeanProgress artifacts to the Hugging Face Hub."""

import json
from pathlib import Path


def validate_dataset(path: Path) -> tuple[list[dict], dict, list[str]]:
    """Read a LeanProgress JSONL dataset and compute its statistics."""
    rows = []
    errors = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(row, dict):
                reason = "row must be an object"
            elif not isinstance(row.get("goal"), str) or not row["goal"]:
                reason = "goal must be a non-empty string"
            elif not isinstance(row.get("tactic"), str) or not row["tactic"]:
                reason = "tactic must be a non-empty string"
            elif (
                isinstance(row.get("steps_remaining"), bool)
                or not isinstance(row.get("steps_remaining"), int)
                or row["steps_remaining"] < 0
            ):
                reason = "steps_remaining must be a non-negative integer"
            elif not isinstance(row.get("prefix", ""), str):
                reason = "prefix must be a string"
            else:
                row.setdefault("prefix", "")
                rows.append(row)
                continue
            errors.append(f"line {line_number}: {reason}")
    steps = [row["steps_remaining"] for row in rows]
    stats = {
        "n_rows": len(rows),
        "steps_remaining": {
            "min": min(steps, default=0),
            "max": max(steps, default=0),
            "mean": sum(steps) / len(steps) if steps else 0.0,
        },
        "rows_with_prefix": sum(bool(row.get("prefix", "")) for row in rows),
    }
    return rows, stats, errors


def build_dataset_card(repo_id: str, stats: dict, license_id: str = "mit") -> str:
    """Render the dataset card for a LeanProgress release.

    The card describes the release as generated for the Hub; it never claims
    the dataset is already published.
    """
    steps = stats["steps_remaining"]
    return f"""---
tags:
- lean
- theorem-proving
- lean4
license: {license_id}
language: en
---

# {repo_id}

LeanProgress dataset: proof states with remaining-step labels for Lean 4
theorem proving. This card was generated for the release; upload it together
with the dataset file.

## Schema

Each row is a JSON object with these fields:

- `goal`: the Lean goal state, as pretty-printed text.
- `prefix`: the proof prefix already established, if any (may be empty).
- `tactic`: a candidate tactic to apply next.
- `steps_remaining`: the number of proof steps remaining after the tactic,
  a non-negative integer.

## Statistics

- Rows: {stats["n_rows"]}
- Rows with a non-empty prefix: {stats["rows_with_prefix"]}
- `steps_remaining`: min {steps["min"]}, max {steps["max"]}, mean {steps["mean"]}

## Load

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```
"""


def build_model_card(
    repo_id: str,
    base_model: str,
    stats: dict | None = None,
    license_id: str = "mit",
) -> str:
    """Render the model card for a LeanProgress regression checkpoint.

    No metric numbers are included: metrics are reported after training, once
    the checkpoint has been evaluated.
    """
    card = f"""---
tags:
- lean
- theorem-proving
library_name: transformers
license: {license_id}
base_model: {base_model}
---

# {repo_id}

LeanProgress regression model: given a Lean goal, a proof prefix, and a
candidate tactic, it predicts `steps_remaining`, the number of proof steps
left after the tactic. The model is a sequence-classification head trained
for regression on top of `{base_model}`.

## Training data

The model is trained on LeanProgress JSONL data with `goal`, `prefix`,
`tactic`, and `steps_remaining` fields, as produced by
`lean_dojo_v2.lean_progress.create_sample_dataset` and the LeanDojo-v2
tracing pipeline.
"""
    if stats is not None:
        steps = stats["steps_remaining"]
        card += f"""
Training rows: {stats["n_rows"]}; `steps_remaining` min {steps["min"]},
max {steps["max"]}, mean {steps["mean"]}.
"""
    card += """
## Metrics

Evaluation metrics are reported after training, once the checkpoint has
been evaluated on a held-out split.
"""
    return card


def _require_hf_token() -> str:
    import os

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Export HF_TOKEN=<hugging-face-token> "
            "before uploading to the Hub, or use --dry-run to validate "
            "and prepare the release without uploading."
        )
    return token


def upload_dataset(
    dataset_path: Path,
    repo_id: str,
    private: bool = False,
    dry_run: bool = False,
    out_dir: Path | None = None,
) -> dict:
    """Validate and upload a LeanProgress dataset to the Hub.

    In dry-run mode the card and a manifest are written to out_dir and
    nothing touches the network. Returns the release manifest.
    """
    dataset_path = Path(dataset_path)
    rows, stats, errors = validate_dataset(dataset_path)
    if errors:
        raise ValueError(
            f"Dataset {dataset_path} failed validation:\n" + "\n".join(errors)
        )
    card = build_dataset_card(repo_id, stats)
    manifest = {
        "repo_id": repo_id,
        "kind": "dataset",
        "n_rows": stats["n_rows"],
        "stats": stats,
        "files": [dataset_path.name, "README.md"],
        "private": private,
        "dry_run": dry_run,
    }
    if dry_run:
        target = Path(out_dir) if out_dir is not None else Path("hf-release-dry-run")
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text(card, encoding="utf-8")
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Dry run: validated {stats['n_rows']} rows, wrote card to {target}")
        return manifest

    token = _require_hf_token()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
    api.upload_file(
        path_or_fileobj=str(dataset_path),
        path_in_repo=dataset_path.name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {stats['n_rows']} rows to {repo_id}")
    return manifest


def upload_model(
    model_dir: Path,
    repo_id: str,
    base_model: str,
    private: bool = False,
    dry_run: bool = False,
    out_dir: Path | None = None,
) -> dict:
    """Validate and upload a LeanProgress checkpoint to the Hub.

    The model directory must contain a config.json. In dry-run mode the card
    and a manifest are written to out_dir and nothing touches the network.
    Returns the release manifest.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Model directory {model_dir} has no config.json. "
            "Point --model-dir at a saved checkpoint directory."
        )
    files = sorted(p.name for p in model_dir.iterdir() if p.is_file())
    card = build_model_card(repo_id, base_model)
    manifest = {
        "repo_id": repo_id,
        "kind": "model",
        "base_model": base_model,
        "files": files + ["README.md"],
        "private": private,
        "dry_run": dry_run,
    }
    if dry_run:
        target = Path(out_dir) if out_dir is not None else Path("hf-release-dry-run")
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text(card, encoding="utf-8")
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Dry run: prepared model card for {repo_id} in {target}")
        return manifest

    token = _require_hf_token()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded model {model_dir} to {repo_id}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the LeanProgress Hub release."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Validate and upload LeanProgress artifacts to the Hugging Face Hub."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--dataset-repo", default="lean-dojo/lean-workbook-plus")
    parser.add_argument("--model-repo", default="lean-dojo/leanprogress-steps")
    parser.add_argument("--base-model", default="bert-base-uncased")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("hf-release-dry-run"))
    args = parser.parse_args(argv)

    try:
        _, _, errors = validate_dataset(args.dataset)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        print(
            f"error: dataset {args.dataset} failed validation:",
            file=sys.stderr,
        )
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 2
    upload_dataset(
        args.dataset,
        args.dataset_repo,
        private=args.private,
        dry_run=args.dry_run,
        out_dir=args.out_dir / "dataset",
    )
    if args.model_dir is not None:
        upload_model(
            args.model_dir,
            args.model_repo,
            args.base_model,
            private=args.private,
            dry_run=args.dry_run,
            out_dir=args.out_dir / "model",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
