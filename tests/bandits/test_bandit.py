import pytest

from bandits.bandit import MultiArmedBandit


@pytest.fixture
def basic_bandit():
    """Bandit with 3 arms: Bernoulli, Gaussian, and Poisson"""
    return MultiArmedBandit([
        ("bernoulli", {"p": 0.2}),
        ("gaussian", {"loc": 1.5, "scale": 0.5}),
        ("poisson", {"mu": 2.0}),
    ], seed=42)


def test_initialization_modes():
    # Test arm-based initialization
    bandit1 = MultiArmedBandit([
        ("uniform", {"loc": 0, "scale": 1}),
        ("uniform", {"loc": 2, "scale": 3}),
    ])
    assert bandit1.n_arms == 2

    # Test distribution+params initialization
    bandit2 = MultiArmedBandit(
        distribution="bernoulli",
        params=[{"p": 0.1}, {"p": 0.5}]
    )
    assert bandit2.n_arms == 2

    # Test conflict error
    with pytest.raises(ValueError):
        MultiArmedBandit([("norm", {})], distribution="gaussian")

    # Test missing params error
    with pytest.raises(ValueError):
        MultiArmedBandit(distribution="gaussian")


def test_pull_behavior(basic_bandit):
    # Test pull returns values
    rewards = [basic_bandit.pull(0) for _ in range(10)]
    assert all(r in {0, 1} for r in rewards)  # Bernoulli

    # Test reproducibility
    basic_bandit.reset(42)
    r1 = basic_bandit.pull(1)
    basic_bandit.reset(42)
    r2 = basic_bandit.pull(1)
    assert r1 == r2


def test_optimal_action_calculation():
    # Create bandit with clear optimal arm
    bandit = MultiArmedBandit([
        ("bernoulli", {"p": 0.4}),
        ("bernoulli", {"p": 0.8}),
        ("bernoulli", {"p": 0.6}),
    ])
    assert bandit.optimal_action_reward == (1, 0.8)

    # Test continuous distributions
    bandit = MultiArmedBandit([
        ("gaussian", {"loc": 1.0, "scale": 0.1}),
        ("gaussian", {"loc": 2.0, "scale": 0.1}),
    ])
    assert bandit.optimal_action_reward[0] == 1


def test_parameter_parsing():
    # Test numeric parameter expansion
    params = MultiArmedBandit._parse_shape_params("bernoulli", 0.3)
    assert params == {"p": 0.3}

    # Test list parameter expansion
    params = MultiArmedBandit._parse_shape_params("norm", [1.5, 0.2])
    assert params == {"loc": 1.5, "scale": 0.2}

    # Test dict parameter pass-through
    params = MultiArmedBandit._parse_shape_params("poisson", {"mu": 3})
    assert params == {"mu": 3}


def test_step_function(basic_bandit):
    # Test basic step without optimal info
    reward = basic_bandit.step(0)
    assert reward in [0, 1]

    # Test step with optimal info
    reward, (optimal_action, optimal_reward) = basic_bandit.step(1, return_optimal=True)
    assert optimal_action == basic_bandit.optimal_action_reward[0]
    assert optimal_reward == basic_bandit.action_values.max()


def test_string_representations(basic_bandit):
    # Test __str__ output
    str_repr = str(basic_bandit)
    assert "Bernoulli(p=0.2)" in str_repr
    assert "Gaussian(loc=1.5, scale=0.5)" in str_repr

    # Test __repr__ recreatability
    repr_str = repr(basic_bandit)
    assert "MultiArmedBandit([" in repr_str
    assert "'bernoulli', {'p': 0.2}" in repr_str


def test_edge_cases():
    # Test single-arm bandit
    bandit = MultiArmedBandit([("uniform", {"loc": 0, "scale": 1})])
    assert bandit.optimal_action_reward == (0, 0.5)

    # Test invalid action
    bandit = MultiArmedBandit(distribution="uniform", params=[{"loc": 0, "scale": 1}])
    with pytest.raises(IndexError):
        bandit.pull(1)
