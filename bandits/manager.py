from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from tqdm.auto import trange


class BanditsManager:
    """
    Manager for single-state environments, such as Multi-Armed Bandits.
    Manages interaction between an environment and learning agent over multiple rounds.

    Tracks agent performance metrics including:
    - Actions taken
    - Rewards received
    - Optimal actions for each round (maybe useful for non-stationary envs)
    - Optimal expected rewards

    Attributes:
        environment: Multi-armed bandit environment
        agent: Bandit learning agent
        n_rounds (int): Number of rounds to run
    """

    def __init__(self, environment, agent, *, n_rounds: int = 1000):
        """
        Initialize experiment manager.

        Args:
            environment: Bandit environment implementing step(action) method
            agent: Learning agent implementing do_action() and observe(reward)
            n_rounds: Number of interaction rounds to run
        """

        self.environment = environment
        self.agent = agent
        self.n_rounds = n_rounds

        self._agent_actions = np.zeros(n_rounds, np.int64)
        self._optimal_actions = np.zeros(n_rounds, np.int64)
        self._rewards = np.zeros(n_rounds, np.float64)
        self._optimal_rewards = np.zeros(n_rounds, np.float64)

    def run(self):
        """
        Execute the bandit experiment.

        Returns:
            dict: Experiment statistics with keys:
                - agent_actions: Array of actions taken (shape: [n_rounds])
                - rewards: Array of received rewards (shape: [n_rounds])
                - optimal_actions: Array of optimal actions (shape: [n_rounds])
                - optimal_rewards: Array of optimal possible rewards (shape: [n_rounds])
        """

        for i in range(self.n_rounds):
            a = self.agent.do_action()
            r, (a_opt, r_exp) = self.environment.step(a, return_optimal=True)
            self.agent.observe(r)

            self._agent_actions[i] = a
            self._optimal_actions[i] = a_opt
            self._rewards[i] = r
            self._optimal_rewards[i] = r_exp

        return {
            "agent_actions": self._agent_actions,
            "rewards": self._rewards,
            "optimal_actions": self._optimal_actions,
            "optimal_rewards": self._optimal_rewards,
        }


def _get_average_performance_thread(environment, agent, n_rounds, n_runs):
    rewards_sum = np.zeros(n_rounds, np.float64)
    optimal_actions_rate = np.zeros(n_rounds, np.int64)
    optimal_rewards = np.zeros(n_rounds, np.float64)

    for _ in range(n_runs):
        environment.reset()
        agent.reset()
        stats = BanditsManager(environment, agent, n_rounds=n_rounds).run()
        rewards_sum += stats["rewards"]
        optimal_actions_rate += (stats["agent_actions"] == stats["optimal_actions"])
        optimal_rewards += stats["optimal_rewards"]

    cumulative_regret = np.cumsum(optimal_rewards - rewards_sum)

    return rewards_sum, optimal_actions_rate, optimal_rewards, cumulative_regret


def get_average_performance(
        environment,
        agent,
        *,
        n_rounds: int = 1000,
        n_runs: int = 100,
        threads: int = 1,
        verbose: bool = False
):
    """
    Calculate average performance metrics over multiple runs.

    Args:
        environment: Bandit environment to use
        agent: Learning agent to test
        n_rounds: Number of rounds per experiment
        n_runs: Total number of experiment runs
        threads: Number of parallel processes to use
        verbose: Show progress bar when using single thread

    Returns:
        dict: Average performance metrics with keys:
            - average_rewards: Mean reward per round (shape: [n_rounds])
            - optimal_actions_rate: Optimal action selection rate (shape: [n_rounds])
            - average_optimal_rewards: Mean optimal rewards (shape: [n_rounds])
            - cumulative_regret: Cumulative regret over time (shape: [n_rounds])
    """

    if threads == 1:
        rewards_sum = np.zeros(n_rounds, dtype=np.float64)
        optimal_actions_rate = np.zeros(n_rounds, dtype=np.int64)
        optimal_rewards = np.zeros(n_rounds, dtype=np.float64)

        iter_ = trange if verbose else range
        for _ in iter_(n_runs):
            environment.reset()
            agent.reset()
            stats = BanditsManager(environment, agent, n_rounds=n_rounds).run()
            rewards_sum += stats["rewards"]
            optimal_actions_rate += (stats["agent_actions"] == stats["optimal_actions"])
            optimal_rewards += stats["optimal_rewards"]

        cumulative_regret = np.cumsum(optimal_rewards - rewards_sum)

        return {
            "average_rewards": rewards_sum / n_runs,
            "optimal_actions_rate": optimal_actions_rate / n_runs,
            "average_optimal_rewards": optimal_rewards / n_runs,
            "cumulative_regret": cumulative_regret / n_runs,
        }
    else:
        n_runs_per_thread = np.ones(threads, dtype=int) * (n_runs // threads)
        n_runs_per_thread[:(n_runs - n_runs_per_thread.sum())] += 1
        assert n_runs_per_thread.sum() == n_runs
        with Pool(threads) as p:
            res_threads = np.array(p.starmap(
                _get_average_performance_thread,
                [(
                    deepcopy(environment),
                    deepcopy(agent),
                    n_rounds,
                    n_runs_per_thread[i]
                ) for i in range(threads)]
            ))
        res_threads = res_threads.sum(axis=0)
        return {
            "average_rewards": res_threads[0] / n_runs,
            "optimal_actions_rate": res_threads[1] / n_runs,
            "average_optimal_rewards": res_threads[2] / n_runs,
            "cumulative_regret": res_threads[3] / n_runs,
        }
