# Multi-Armed Bandits

This repository provides tools for exploring multi-armed bandit (MAB) problems.
It includes implementations of decision-making strategies such as Greedy, ε-Greedy, UCB-1, and more.
The library allows customization of bandit parameters, running experiments, and visualizing results.

## Key Features

- **Multiple Reward Distributions**: Bernoulli, Gaussian, Uniform, Binomial, and more.
- **Customizable Bandit Arms**: Configure parameters for each arm (e.g., mean, variance).
- **Strategy Implementations**:
  - Greedy
  - ε-Greedy
  - εₙ-Greedy (Epsilon-decreasing greedy)
  - UCB-1 (Upper Confidence Bound)
  - Random
- **Visualization Tools**: Reward distributions, cumulative rewards, regret curves.
- **Parallel Experiment Execution**: Speed up large-scale comparisons using multi-threading.

## Quick Example

```python
from bandits.bandit import MultiArmedBandit
from bandits.agent import EpsilonGreedy
from bandits.manager import BanditsManager

# Create a bandit with two Bernoulli arms (p=0.3 and p=0.7)
bandit = MultiArmedBandit(distribution="bernoulli", params=[0.3, 0.7], seed=42)

# Initialize an ε-Greedy agent (ε=0.1)
agent = EpsilonGreedy(n_actions=2, epsilon=0.1)

# Run 1000 rounds of interaction
manager = BanditsManager(environment=bandit, agent=agent, n_rounds=1000)
results = manager.run()

# Output results
print("Total reward:", results["rewards"].sum())
print("Optimal action rate (%):", results["optimal_actions_rate"][-1] * 100)
```

## Experiments and Analysis

The `MultiArmedBandits.ipynb` notebook demonstrates:

- **Strategy comparisons** across different bandit types (Bernoulli, Gaussian, Uniform).
- **Parameter sensitivity analysis** (e.g., ε values for ε-Greedy).
- **Regret curves** and cumulative reward visualizations.
- **Adaptive strategies** (e.g., εₙ-Greedy with decaying exploration).
