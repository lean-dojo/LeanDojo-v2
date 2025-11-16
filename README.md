# LeanDojo-v2

LeanDojo-v2 is an end-to-end framework for training, evaluating, and deploying AI-assisted theorem provers for Lean 4. It combines repository tracing, lifelong dataset management, retrieval-augmented agents, Hugging Face fine-tuning, and external inference APIs into one toolkit.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Repository Layout](#repository-layout)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Environment Setup](#environment-setup)
7. [Quickstart](#quickstart)
8. [Working with Agents and Trainers](#working-with-agents-and-trainers)
9. [Tracing and Dataset Generation](#tracing-and-dataset-generation)
10. [External APIs and LeanCopilot](#external-apis-and-leancopilot)
11. [Testing](#testing)
12. [Troubleshooting & Tips](#troubleshooting--tips)
13. [Contributing](#contributing)
14. [License](#license)

---

## Overview

LeanDojo-v2 extends the original LeanDojo stack with the LeanAgent lifelong learning pipeline. It automates the entire loop of:

1. Cloning Lean repositories (GitHub or local) and tracing them with Lean instrumentation.
2. Storing structured theorem information in a dynamic database.
3. Training agent policies with supervised fine-tuning (SFT), GRPO-style RL, or retrieval objectives.
4. Driving Pantograph-based provers to fill in sorrys or verify solutions.
5. Surfacing models through LeanCopilot and HTTP APIs for IDE integrations.

The codebase is modular: you can reuse the tracing pipeline without the agents, swap in custom trainers, or stand up your own inference service via the external API layer.

---

## Key Features

- **Unified Agent Abstractions**: `BaseAgent` orchestrates repository setup, training, and proving. Concrete implementations (`HFAgent`, `LeanAgent`, and `ExternalAgent`) tailor the workflow to Hugging Face models, retrieval-based provers, or REST-backed models.
- **Powerful Trainers**: `SFTTrainer`, `GRPOTrainer`, and `RetrievalTrainer` cover LoRA-enabled supervised fine-tuning, group-relative policy optimization, and retriever-only curriculum learning.
- **Multi-Modal Provers**: `HFProver`, `RetrievalProver`, and `ExternalProver` run on top of Pantograph’s Lean RPC server to search for tactics, generate whole proofs, or delegate to custom models.
- **Lean Tracing Pipeline**: `lean_dojo_v2/lean_dojo` includes the Lean 4 instrumentation (`ExtractData.lean`) and Python utilities to trace commits, normalize ASTs, and cache proof states.
- **Dynamic Repository Database**: `lean_agent.database` tracks repositories, theorems, curriculum difficulty, and sorry status, enabling lifelong training schedules.
- **External API + LeanCopilot**: The `external_api` folder exposes HTTP endpoints (FastAPI + uvicorn) and Lean frontend snippets so you can query LLMs from Lean editors.
- **Utilities for Reproducibility**: Shared helpers manage Git interactions, filesystem layout, environment validation, and experiment tracking.

---

## Repository Layout

| Path | Description |
|------|-------------|
| `lean_dojo_v2/agent/` | Base class plus `HFAgent`, `LeanAgent`, and helpers to manage repositories and provers. |
| `lean_dojo_v2/trainer/` | SFT, GRPO, and retrieval trainers with Hugging Face + DeepSpeed integration. |
| `lean_dojo_v2/prover/` | Pantograph-based prover implementations (HF, retrieval, external). |
| `lean_dojo_v2/lean_dojo/` | Lean tracing, dataset generation, caching, and AST utilities. |
| `lean_dojo_v2/lean_agent/` | Lifelong learning pipeline (configs, database, retrieval stack, generator). |
| `lean_dojo_v2/external_api/` | LeanCopilot code (Lean + Python server) to query external models. |
| `lean_dojo_v2/utils/` | Shared helpers for Git, filesystem operations, and constants. |
| `lean_dojo_v2/tests/` | Pytest regression suite (`test_dojo.py`). |

For deeper documentation on the lifelong learning component, see `lean_dojo_v2/lean_agent/README.md`.

---

## Requirements

- Python ≥ 3.11.
- CUDA-capable GPU for training and inference (tested with CUDA 12.6).
- Git ≥ 2.25 and `wget`.
- [elan](https://github.com/leanprover/elan) Lean toolchain to trace repositories locally.
- Adequate disk space for the `raid/` working directory (datasets, checkpoints, traces).

Python dependencies are declared in `pyproject.toml` and include PyTorch, PyTorch Lightning, Transformers, DeepSpeed, TRL, PEFT, and more.

---

## Installation

### Option 1: From PyPI

```sh
# Install the core package
pip install lean-dojo-v2

# Pantograph is required for Lean RPC
pip install git+https://github.com/stanford-centaur/PyPantograph

# Install a CUDA-enabled torch build (adjust the index URL for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Option 2: From Source (development)

```sh
git clone https://github.com/lean-dojo/LeanDojo-v2.git
cd LeanDojo-v2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
pip install git+https://github.com/stanford-centaur/PyPantograph
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> Tip: You can use [uv](https://github.com/astral-sh/uv) (`uv pip install lean-dojo-v2`) as an alternative Python package manager.

---

## Environment Setup

1. **GitHub Access Token (required)**  
   The tracing pipeline calls the GitHub API extensively. Create a personal access token and export it before running any agent:
   ```sh
   export GITHUB_ACCESS_TOKEN=<your-token>
   ```

2. **Hugging Face Token (optional but needed for gated models)**  
   ```sh
   export HF_TOKEN=<your-hf-token>
   ```

3. **Working directories**  
   By default all datasets, caches, and checkpoints live under `<repo>/raid`. Change the layout by editing `lean_dojo_v2/utils/constants.py` or by pointing `RAID_DIR` to faster storage.

4. **Lean toolchains**  
   Ensure `elan` is configured and Lean 4 (e.g., `leanprover/lean4:nightly`) is available on your `$PATH`. The tracing scripts look under `~/.elan/toolchains/`.

---

## Quickstart

```python
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

trainer = SFTTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir="outputs-deepseek",
    epochs_per_repo=1,
    batch_size=2,
    lr=2e-5,
)

agent = HFAgent(trainer=trainer)
agent.setup_github_repository(url=url, commit=commit)
agent.train()
agent.prove()
```

This example:
1. Downloads and traces the target Lean repository + commit.
2. Builds a supervised dataset from sorry theorems.
3. Fine-tunes the specified Hugging Face model (optionally with LoRA).
4. Launches an `HFProver` backed by Pantograph to search for proofs.

---

## Working with Agents and Trainers

### Supervised Fine-Tuning (`SFTTrainer`)
- Accepts any Hugging Face causal LM identifier.
- Supports LoRA by passing a `peft.LoraConfig`.
- Key arguments: `epochs_per_repo`, `batch_size`, `max_seq_len`, `lr`, `warmup_steps`, `gradient_checkpointing`.
- Produces checkpoints under `output_dir` that the `HFProver` consumes.

### GRPO Trainer (`GRPOTrainer`)
- Implements Group Relative Policy Optimization for reinforcement-style refinement.
- Accepts `reference_model`, `reward_weights`, and `kl_beta` settings.
- Useful for improving search policies on curated theorem batches.

### Retrieval Trainer & LeanAgent
- `RetrievalTrainer` trains the dense retriever that scores prior proofs.
- `LeanAgent` wraps the trainer, maintains repository curricula, and couples it with `RetrievalProver`.
- Run via:
  ```sh
  python -m lean_dojo_v2.lean_agent
  ```
  then customize `TrainingConfig`/`ProverConfig` to point to your checkpoints and Fisher matrices.

Each agent inherits `BaseAgent`, so you can implement your own by overriding `_get_build_deps()` and `_setup_prover()` to register new trainer/prover pairs.

---

## Tracing and Dataset Generation

The `lean_dojo_v2/lean_dojo/data_extraction` package powers repository tracing:

- `lean.py` clones repositories (GitHub, remote, or local), validates Lean versions, and normalizes URLs.
- `trace.py` drives Lean with the custom `ExtractData.lean` instrumented module to capture theorem states.
- `dataset.py` converts traced files to JSONL datasets ready for trainers.
- `cache.py` memoizes repository metadata to avoid redundant downloads.
- `traced_data.py` exposes typed wrappers for traced AST nodes and sorrys.

Typical usage:

```python
from lean_dojo_v2.lean_dojo.data_extraction.dataset import generate_benchmark

generate_benchmark(
    repos=[("https://github.com/durant42040/lean4-example", "005de00d03f1aaa32cb2923d5e3cbaf0b954a192")],
    build_deps=True,
    output_dir="raid/data",
)
```

The generated artifacts flow into the `DynamicDatabase`, which keeps repositories sorted by difficulty and appends new sorrys without retracing everything.

---

## External APIs and LeanCopilot

`lean_dojo_v2/external_api` contains Lean and Python code to expose models through LeanCopilot:

- `LeanCopilot.lean` registers RPC endpoints inside Lean.
- `python/server.py` hosts a FastAPI service with adapters for Anthropic, OpenAI, Google Generative AI, vLLM, and custom HF models.
- Start the service with:
  ```sh
  cd lean_dojo_v2/external_api/python
  pip install -r requirements.txt
  uvicorn server:app --port 23337
  ```
- Point your Lean client to the running server to interactively request tactics, proofs, or completions from external models.

### LeanProgress Step-Prediction Workflow

- Generate a JSONL dataset with remaining-step targets (or replace it with your own LeanProgress export):
  ```sh
  python examples/leanprogress/create_sample_dataset.py --output raid/data/sample_leanprogress_dataset.jsonl
  ```
- Fine-tune a regression head that predicts `steps_remaining`:
  ```sh
  python examples/leanprogress/train_steps_model.py \
    --dataset raid/data/sample_leanprogress_dataset.jsonl \
    --output-dir raid/checkpoints/leanprogress_steps \
    --model-name bert-base-uncased
  ```
- Tell the LeanCopilot server where to find the checkpoint by exporting:
  ```sh
  export LEANPROGRESS_MODEL=raid/checkpoints/leanprogress_steps
  uvicorn server:app --port 23337
  ```
- Add `use_reward=true` when calling `/generate`. Each output now includes `steps_remaining` and a reward value (currently `-steps_remaining`) so agents can minimize proof length.

---

## Testing

We use `pytest` for regression coverage.

```sh
pip install -e .[dev]          # make sure dev extras like pytest/trl are present
export GITHUB_ACCESS_TOKEN=<token>
export HF_TOKEN=<hf-token>     # only required for tests touching HF APIs
pytest
```

Tests currently cover two areas:
- `lean_dojo_v2/tests/test_dojo.py` spins up the full tracing + Lean agent flow on a tiny public repository. It needs valid GitHub and Hugging Face tokens, plus a working Lean toolchain/Pantograph installation.
- `tests/test_leanprogress_examples.py` exercises the LeanProgress dataset helper and regression tokenizer logic. These are pure Python tests with no external dependencies.

If you only want to run the examples test suite (no network calls), target it directly:

```sh
pytest tests/test_leanprogress_examples.py
```

The Lean tracing tests clone repositories and build Lean projects, so make sure `elan` is installed and the `raid/` directory has sufficient space.

---

## Troubleshooting & Tips

- **401 Bad Credentials / rate limits**: Ensure `GITHUB_ACCESS_TOKEN` is exported and has `repo` + `read:org` scopes.
- **Lean tracing failures**: Confirm that the repo’s Lean version exists locally (`elan toolchain install <version>`).
- **Missing CUDA libraries**: Install the PyTorch wheel that matches your driver and CUDA version.
- **Dataset location**: The default `raid/` directory can grow large. Point it to high-throughput storage or use symlinks.
- **Pantograph errors**: Reinstall Pantograph from source (`pip install git+https://github.com/stanford-centaur/PyPantograph`) whenever Lean upstream changes.

---

## Contributing

Issues and pull requests are welcome! Please:

1. Open an issue describing the bug or feature.
2. Run formatters (`black`, `isort`) and `pytest` before submitting.
3. Mention if your change touches Lean tracing files so reviewers can re-generate artifacts.

---

## License

LeanDojo-v2 is released under the MIT License. See `LICENSE` for details.
