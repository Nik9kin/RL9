from typing import Any, Callable

import numpy as np
from IPython.display import clear_output

from games.base import BasePlayer, BaseState


class IOPlayer(BasePlayer):
    def __init__(self, from_str: Callable[[str], Any], *, clear_outputs: bool = False):
        self.from_str = from_str
        self.clear_outputs = clear_outputs

    def do_action(self, state: BaseState) -> Any:
        if self.clear_outputs:
            clear_output()
        print("Current state:")
        print(state, flush=True)
        action = input("Your action: ")
        return self.from_str(action)

    def observe(self, reward: int, final_state: BaseState | None = None) -> None:
        if self.clear_outputs:
            clear_output()
        if reward == 1:
            print("You are win!")
        elif reward == 0:
            print("It`s a draw.")
        else:
            print("You are lose :(")
        if final_state is not None:
            print("Final state:")
            print(final_state)


class RandomPlayer(BasePlayer):
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def do_action(self, state: BaseState) -> Any:
        return self._rng.choice(state.actions)
