from typing import Any

import numpy as np


def select_random_optimal_action(
        action_values: np.ndarray,
        actions: list[Any] | None = None,
        **kwargs
):
    action_ind = np.random.choice(
        np.nonzero(np.isclose(action_values, action_values.max(), **kwargs))[0]
    )
    if actions:
        return actions[action_ind]
    return action_ind
