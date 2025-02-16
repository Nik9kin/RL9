import pytest
import numpy as np

from bandits.manager import BanditsManager, get_average_performance


class MockEnvironment:
    """Deterministic test environment with 2 arms"""

    def __init__(self):
        self.optimal = (0, 1.0)  # (action, reward)

    def reset(self):
        pass

    def step(self, action, return_optimal=False):
        reward = 1.0 if action == 0 else 0.5
        return (reward, self.optimal) if return_optimal else reward


class MockAgent:
    """Deterministic test agent that always chooses action 0"""

    def __init__(self):
        self.n_actions = 2

    def reset(self):
        pass

    def do_action(self):
        return 0

    def observe(self, reward):
        pass


@pytest.fixture
def mock_setup():
    return MockEnvironment(), MockAgent()


def test_bandits_manager_basic(mock_setup):
    env, agent = mock_setup
    manager = BanditsManager(env, agent, n_rounds=10)
    results = manager.run()

    assert len(results["agent_actions"]) == 10
    assert np.all(results["agent_actions"] == 0)
    assert np.all(results["rewards"] == 1.0)
    assert np.all(results["optimal_actions"] == 0)
    assert np.all(results["optimal_rewards"] == 1.0)


def test_single_thread_performance(mock_setup):
    env, agent = mock_setup
    results = get_average_performance(env, agent, n_rounds=10, n_runs=5)

    assert results["average_rewards"].shape == (10,)
    assert np.allclose(results["average_rewards"], 1.0)
    assert np.allclose(results["optimal_actions_rate"], 1.0)
    assert np.allclose(results["cumulative_regret"], 0.0)


def test_multi_thread_performance(mock_setup):
    env, agent = mock_setup
    results = get_average_performance(env, agent, n_rounds=10, n_runs=4, threads=2)

    assert results["average_rewards"].shape == (10,)
    assert np.allclose(results["average_rewards"], 1.0)
    assert np.allclose(results["optimal_actions_rate"], 1.0)


def test_regret_calculation():
    class SuboptimalEnv(MockEnvironment):
        def step(self, action, return_optimal=False):
            return (0.5, (0, 1.0)) if return_optimal else 0.5

    env = SuboptimalEnv()
    agent = MockAgent()  # Always gets 0.5 reward

    results = get_average_performance(env, agent, n_rounds=5, n_runs=2)
    expected_regret = np.cumsum([0.5] * 5)
    assert np.allclose(results["cumulative_regret"], expected_regret)
