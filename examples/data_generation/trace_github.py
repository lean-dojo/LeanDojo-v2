"""
Example script for generating dataset from a Lean 4 GitHub repository. 
The data is saved at <RAID_DIR>/<DATA_DIR>/<repo.name>_<repo.commit>.
e.g. LeanDojo-v2/raid/data/lean4-example_3e23ab0bfdcfdbd5b11ab53c2cd8b5d16492e9c2

Usage: python examples/data_generation/trace_github.py
"""

from lean_dojo_v2.database import DynamicDatabase


def main() -> None:
    url = "https://github.com/durant42040/lean4-example"
    commit = "3e23ab0bfdcfdbd5b11ab53c2cd8b5d16492e9c2"

    database = DynamicDatabase()

    database.trace_repository(
        url=url,
        commit=commit,
        build_deps=False,
    )


if __name__ == "__main__":
    main()
