import numpy as np
import scipy.stats as ss

DISTRIBUTIONS = {
    "bernoulli": ss.bernoulli,
    "binomial": ss.binom,
    "cauchy": ss.cauchy,
    "exponential": ss.expon,
    "gaussian": ss.norm,
    "geom": ss.geom,
    "laplace": ss.laplace,
    "norm": ss.norm,
    "poisson": ss.poisson,
    "uniform": ss.uniform,
}


class MultiArmedBandit:
    def __init__(
            self,
            arms = None,
            *,
            distr_name: str | None = None,
            params = None,
            seed: int | None = None
    ):
        if arms is not None:
            if distr_name is not None or params is not None:
                raise ValueError(
                    "Cannot specify both 'arms' and any of 'distr_name' or 'params'"
                )
        elif distr_name is None or params is None:
            raise ValueError("Need to specify 'arms' or 'distr_name' and 'params'")
        else:
            if distr_name not in DISTRIBUTIONS:
                raise ValueError(f"Unsupported distr_name {distr_name}")
            distr = DISTRIBUTIONS[distr_name]
            arms = [(distr_name, self._parse_shape_params(distr, ps)) for ps in params]

        self.arms = arms
        self.seed = seed

        self.n_arms = len(arms)
        self.rng = None

        self.reset(seed=self.seed)

    def reset(self, *, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def pull(self, action: int):
        if not (0 <= action < self.n_arms):
            raise ValueError(f"'action' must be in range [0, {self.n_arms})")

        distr_name, params = self.arms[action]
        return DISTRIBUTIONS[distr_name].rvs(**params, random_state=self.rng)

    def step(self, action: int, return_optimal: bool = False):
        if return_optimal:
            a_opt, r_exp = self.optimal_action_reward
            return self.pull(action), (a_opt, r_exp)
        else:
            return self.pull(action)

    @property
    def optimal_action_reward(self):
        means = [DISTRIBUTIONS[distr_name].mean(**params) for distr_name, params in self.arms]
        a_opt = np.argmax(means)
        r_exp = means[a_opt]
        return a_opt, r_exp

    @staticmethod
    def _parse_shape_params(
            distr: ss.rv_continuous | ss.rv_discrete,
            params: int | float | tuple | list
    ) -> dict[str, int | float]:
        if isinstance(params, (int, float)):
            params = (params,)

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
        MultiArmedBandit(distr_name="bernoulli", params=[0.9, 0.6]),
        MultiArmedBandit(distr_name="binomial", params=[(10, 0.4), (5, 0.8), (1, 1.0, 4), (3, 0.0, 1.5)]),
        MultiArmedBandit(distr_name="cauchy", params=[-1, 0, (1, 2)]),
        MultiArmedBandit(distr_name="exponential", params=[-1, 0, 1])
    ]
    for b in bandits:
        for a in range(b.n_arms):
            print(b.pull(a), end=' ')
        print()
    # for distr in DISTRIBUTIONS.values():
    #     print(distr.name, distr.shapes)
