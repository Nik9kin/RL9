from collections.abc import Collection
from numbers import Real
from typing import Literal

import numpy as np
import scipy.stats as ss


DistributionName = Literal[
    "bernoulli",
    "binom",
    "binomial",
    "cauchy",
    "expon",
    "exponential",
    "gaussian",
    "geom",
    "laplace",
    "norm",
    "poisson",
    "uniform",
]
DistributionInfo = tuple[DistributionName, dict[str, Real]]


DISTRIBUTIONS = {
    "bernoulli": ss.bernoulli,
    "binom": ss.binom,
    "binomial": ss.binom,
    "cauchy": ss.cauchy,
    "expon": ss.expon,
    "exponential": ss.expon,
    "gaussian": ss.norm,
    "geom": ss.geom,
    "laplace": ss.laplace,
    "norm": ss.norm,
    "poisson": ss.poisson,
    "uniform": ss.uniform,
}


class MultiArmedBandit:
    """
    A configurable multi-armed bandit environment with parametric distributions.

    Can be initialized either with:
    - A collection of pre-configured arms (distribution + parameters), or
    - A common distribution type with different parameters for each arm

    Attributes:
        n_arms (int): Number of arms in the bandit
        seed (int | None): Random seed for reproducibility
        action_values (np.ndarray): Array of expected rewards for each arm
        optimal_action_reward (tuple[int, float]): Tuple containing:
            - Index of the optimal arm
            - Expected reward of the optimal arm
    """

    def __init__(
            self,
            arms: Collection[DistributionInfo] | None = None,
            *,
            distribution: DistributionName | None = None,
            params: Collection[Real | Collection[Real] | dict[str, Real]] | None = None,
            seed: int | None = None
    ):
        """
        Initialize the multi-armed bandit.

        Args:
            arms: Collection of arm configurations as (distribution, parameters) tuples.
                Mutually exclusive with distribution/params.
            distribution: Common distribution type for all arms. Requires params.
            params: Collection of parameter sets for the common distribution.
                Each element can be a number, collection of numbers, or parameter dict.
            seed: Random seed for reproducible arm pulls.

        Raises:
            ValueError: On conflicting initialization parameters or unsupported distributions.
        """

        if arms is not None:
            if distribution is not None or params is not None:
                raise ValueError(
                    "Cannot specify both 'arms' and any of 'distribution' or 'params'"
                )
        elif distribution is None or params is None:
            raise ValueError("Need to specify 'arms' or 'distribution' and 'params'")
        else:
            if distribution not in DISTRIBUTIONS:
                raise ValueError(f"Unsupported distribution '{distribution}'")
            arms = [
                (distribution, self._parse_shape_params(distribution, ps))
                for ps in params
            ]

        self._arms = [DISTRIBUTIONS[distr_name](**ps) for distr_name, ps in arms]
        self._arm_descriptions = arms

        self.seed = seed
        self.n_arms = len(self._arms)
        self._rng = None

        self.reset(self.seed)

    def __str__(self) -> str:
        arms_str = ",\n".join(
            f"    {distr_name.capitalize()}({', '.join(f'{k}={v}' for k, v in ps.items())})"
            for distr_name, ps in self._arm_descriptions
        )
        return f"MultiArmedBandit(\n{arms_str}\n)"

    def __repr__(self) -> str:
        arms_str = ",\n".join(
            f"    ('{distr_name}', {{{', '.join(f'\'{k}\': {v}' for k, v in ps.items())}}})"
            for distr_name, ps in self._arm_descriptions
        )
        return f"MultiArmedBandit([\n{arms_str}\n])"

    def reset(self, seed: int | None = None):
        """
        Reset the bandit's random number generator.

        Args:
            seed (int): New random seed. Use None for non-reproducible randomness.
        """
        self._rng = np.random.default_rng(seed)

    def pull(self, action: int):
        """
        Pull the specified arm and get the reward.

        Args:
            action (int): Index of the arm to pull (0-based)

        Returns:
            float: Sampled reward from the selected arm

        Raises:
            IndexError: If action is out of bounds
        """
        return self._arms[action].rvs(random_state=self._rng)

    def step(self, action: int, *, return_optimal: bool = False):
        """
        Pull an arm and optionally return optimal action info.

        Args:
            action (int): Index of the arm to pull
            return_optimal (bool): Whether to return optimal action information (default `False`)

        Returns:
            float: If return_optimal=False
            tuple[float, tuple[int, float]]: If return_optimal=True, returns
                (reward, (optimal_action, optimal_reward))
        """

        reward = self.pull(action)
        if return_optimal:
            return reward, self.optimal_action_reward
        return reward

    @property
    def action_values(self):
        """np.ndarray: Expected rewards for each arm (distribution means)."""
        return np.array([distr.mean() for distr in self._arms])

    @property
    def optimal_action_reward(self):
        """tuple[int, float]: Optimal arm index and its expected reward."""

        means = self.action_values
        a_opt = np.argmax(means)
        r_exp = means[a_opt]
        return a_opt, r_exp

    @staticmethod
    def _parse_shape_params(
            distribution_name: DistributionName,
            params: Real | Collection[Real] | dict[str, Real],
    ) -> dict[str, Real]:
        if isinstance(params, dict):
            return params

        if isinstance(params, Real):
            params = [params]

        distr = DISTRIBUTIONS[distribution_name]
        shape_kwrds = distr.shapes.split(', ') if distr.shapes is not None else []
        if isinstance(distr, ss.rv_continuous):
            shape_kwrds += ["loc", "scale"]
        else:
            shape_kwrds += ["loc"]

        if len(params) > len(shape_kwrds):
            raise ValueError(f"Too much shape parameters ({len(params)})")

        return {k: v for k, v in zip(shape_kwrds, params)}


class NonStationaryBandit(MultiArmedBandit):
    pass


class AdversarialBandit(NonStationaryBandit):
    pass


if __name__ == '__main__':
    bandits = [
        MultiArmedBandit(distribution="bernoulli", params=[0.9, 0.6, 0.0, 1.0]),
        MultiArmedBandit(
            distribution="binomial",
            params=[(5, 0.8), {"n": 1, "p": 1.0, "loc": 4}, [3, 0.0, -2]]
        ),
        MultiArmedBandit(distribution="cauchy", params=[0, (1, 2), (), [], {"scale": 3.5}]),
        MultiArmedBandit(distribution="gaussian", params=[()] * 3),
        MultiArmedBandit(
            [
                ("bernoulli", {"p": 0.5}),
                ("binomial", {"n": 5, "p": 0.1}),
                ("norm", {"loc": 0.5, "scale": 1.0}),
                ("uniform", {}),
            ],
        )
    ]
    for b in bandits:
        print(b)
        print(repr(b))
        print([b.pull(a) for a in range(b.n_arms)])
        print([b.step(a) for a in range(b.n_arms)])
        print(b.action_values)
        print(b.optimal_action_reward)
