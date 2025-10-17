"""
Usage: python examples/lean_agent.py
"""

from lean_dojo_v2.agent.lean_agent import LeanAgent

url = "https://github.com/durant42040/lean4-example"
commit = "b14fef0ceca29a65bc3122bf730406b33c7effe5"

agent = LeanAgent()
agent.setup_github_repository(url=url, commit=commit)
agent.train()
agent.prove()
