"""
Usage: python examples/external_api.py

Example script for using the Hugging Face API to prove sorry theorems in a GitHub repository.
"""

from lean_dojo_v2.agent import ExternalAgent


def main() -> None:
    url = "https://github.com/durant42040/lean4-example"
    commit = "3e23ab0bfdcfdbd5b11ab53c2cd8b5d16492e9c2"

    agent = ExternalAgent(model_name="deepseek-ai/DeepSeek-Prover-V2-671B:novita")
    agent.setup_github_repository(url=url, commit=commit)
    agent.prove(whole_proof=True)


if __name__ == "__main__":
    main()
