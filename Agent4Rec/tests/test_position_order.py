import importlib.util
import random
from pathlib import Path

from simulation.position_order import deterministic_page_permutation


def load_simuser_runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "PQA/SimUSER/MovieLens/scripts/run_vanilla_common.py"
    )
    spec = importlib.util.spec_from_file_location("simuser_vanilla_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_permutation_is_cross_runner_deterministic():
    simuser = load_simuser_runner()
    for seed in (0, 20261121, 20261122):
        for user_id in (0, 17, 999):
            for page in (1, 3, 5):
                agent = deterministic_page_permutation(4, seed, user_id, page)
                other = simuser.deterministic_page_permutation(4, seed, user_id, page)
                assert agent == other
                assert sorted(agent) == [0, 1, 2, 3]


def test_permutation_does_not_modify_global_rng():
    random.seed(42)
    expected = random.random()
    random.seed(42)
    deterministic_page_permutation(4, 20261121, 0, 1)
    assert random.random() == expected
