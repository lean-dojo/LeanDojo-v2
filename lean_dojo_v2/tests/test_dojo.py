from lean_dojo_v2 import BaseAgent

class DumbAgent(BaseAgent):
    __test__ = False
    def _get_build_deps(self) -> bool:
        return False
    def _setup_prover(self) -> None:
        pass

def test_tracing():
    url = "https://github.com/durant42040/lean4-example"
    commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"
    agent = DumbAgent()
    agent.setup_github_repository(url=url, commit=commit)