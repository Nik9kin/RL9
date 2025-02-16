import pytest
import numpy as np

from bandits.agent import BaseAgent, EpsilonGreedy, Greedy, Random, UCB1


def test_base_agent_abstract():
    with pytest.raises(TypeError):
        BaseAgent(3)


def test_epsilon_greedy_exploring_start():
    agent = EpsilonGreedy(3, epsilon=0.5, exploring_cycles=2, seed=42)
    actions = [agent.do_action() for _ in range(6)]  # 2 cycles * 3 actions
    assert actions == [0, 1, 2, 0, 1, 2]


def test_epsilon_greedy_epsilon_dynamic():
    def epsilon_fn(t):
        return 0.5 if t < 5 else 0.1
    agent = EpsilonGreedy(2, epsilon=epsilon_fn, seed=42)
    agent._t = 3  # t + 1 = 4 < 5 -> epsilon=0.5
    assert agent.epsilon == 0.5
    agent._t = 5  # t + 1 = 6 -> epsilon=0.1
    assert agent.epsilon == 0.1


def test_epsilon_greedy_exploit():
    agent = EpsilonGreedy(2, epsilon=0.0, seed=42, exploring_cycles=1)
    agent.do_action()  # Action 0
    agent.observe(1.0)
    agent.do_action()  # Action 1
    agent.observe(3.0)
    # Now, exploit should choose action 1
    assert agent.do_action() == 1


def test_greedy_agent():
    agent = Greedy(2, exploring_cycles=1)
    agent.do_action()  # Explore action 0
    agent.observe(2.0)
    agent.do_action()  # Explore action 1
    agent.observe(1.0)
    # Now, should exploit action 0 (higher value)
    assert agent.do_action() == 0


def test_random_agent():
    agent = Random(10, seed=42)
    actions = [agent.do_action() for _ in range(10)]
    # Since exploring_cycles=0, all actions are random
    assert len(set(actions)) > 1  # Very likely to have multiple actions


def test_ucb1_exploring_start():
    agent = UCB1(3, c=1.0, exploring_cycles=2)
    actions = [agent.do_action() for _ in range(6)]  # 2 cycles * 3 actions
    assert actions == [0, 1, 2, 0, 1, 2]


def test_ucb1_selection():
    agent = UCB1(2, c=2.0, exploring_cycles=1)
    # Exploring phase: actions 0 and 1
    agent.do_action()
    agent.observe(1.0)
    agent.do_action()
    agent.observe(1.5)
    # Next action should use UCB. Values are [1.0, 1.5], attempts [1,1], t=2
    # UCB terms: 2*sqrt(ln(2)/1) ≈ 1.665 → totals ≈ 2.665 and 3.165 → pick 1
    assert agent.do_action() == 1


def test_reset():
    agent = EpsilonGreedy(2, epsilon=0.5, seed=42)
    agent.do_action()
    agent.observe(2.0)
    agent.reset()
    assert np.all(agent.value_estimates == 0)
    assert agent._t == 0


def test_observe_updates():
    agent = EpsilonGreedy(2, epsilon=0.0)
    agent.do_action()  # Action 0
    agent.observe(3.0)
    assert agent.value_estimates[0] == 3.0
    agent.do_action()  # Action 1 (exploring)
    agent.observe(4.0)
    assert agent.value_estimates[1] == 4.0
