from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from tqdm.auto import trange


class BanditsManager:
    """
        Manager for single-state environments, such as Multi-Armed Bandits
    """
    def __init__(self, environment, agent, *, n_rounds: int = 1000):
        self.environment = environment
        self.agent = agent
        self.n_rounds = n_rounds

        self._agent_actions = []
        self._optimal_actions = []
        self._rewards = []
        self._optimal_rewards = []

    def run(self):
        for i in range(self.n_rounds):
            a = self.agent.do_action()
            r, (a_opt, r_exp) = self.environment.step(a, return_optimal=True)
            self.agent.observe(r)

            self._agent_actions.append(a)
            self._optimal_actions.append(a_opt)
            self._rewards.append(r)
            self._optimal_rewards.append(r_exp)

        return {
            "agent_actions": np.array(self._agent_actions),
            "optimal_actions": np.array(self._optimal_actions),
            "rewards": np.array(self._rewards),
            "optimal_rewards": np.array(self._optimal_rewards)
        }


def _get_average_performance_thread(environment, agent, n_rounds, n_runs):
    rewards_sum = np.zeros(n_rounds, dtype=np.float64)
    optimal_actions_rate = np.zeros(n_rounds, dtype=np.int64)
    optimal_rewards = np.zeros(n_rounds, dtype=np.float64)

    for _ in range(n_runs):
        environment.reset()
        agent.reset()
        stats = BanditsManager(environment, agent, n_rounds=n_rounds).run()
        rewards_sum += stats["rewards"]
        optimal_actions_rate += (stats["agent_actions"] == stats["optimal_actions"])
        optimal_rewards += stats["optimal_rewards"]

    return rewards_sum, optimal_actions_rate, optimal_rewards


def get_average_performance(
        environment,
        agent,
        *,
        n_rounds: int = 1000,
        n_runs: int = 100,
        threads: int = 1,
        verbose: bool = False
):
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

        return {
            "average_rewards": rewards_sum / n_runs,
            "optimal_actions_rate": optimal_actions_rate / n_runs,
            "average_optimal_rewards": optimal_rewards / n_runs
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
            "average_optimal_rewards": res_threads[2] / n_runs
        }
