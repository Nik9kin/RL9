from typing import Any

import numpy as np

from game.core.base import BasePlayer, BaseGameState


class RandomPlayer(BasePlayer):
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def do_action(self, state: BaseGameState) -> Any:
        # direct usage of self._rng.choice(actions) may lead to unexpected casting to ndarray
        actions = state.actions
        idx = self._rng.choice(len(actions))
        return actions[idx]
