from typing import Any, Callable

from IPython.display import clear_output

from game.core.base import BaseGameState, BasePlayer


class IOPlayer(BasePlayer):
    def __init__(self, from_str: Callable[[str], Any] | str, *, clear_outputs: bool = False):
        self.from_str: Callable[[str], Any]
        if from_str == 'int':
            self.from_str = int
        elif from_str == 'list[int]':
            self.from_str = lambda s: list(map(int, s.split()))
        elif callable(from_str):
            self.from_str = from_str
        else:
            raise ValueError('"from_str" not recognized')
        self.clear_outputs = clear_outputs

    def do_action(self, state: BaseGameState) -> Any:
        if self.clear_outputs:
            clear_output()
        print("Current state:")
        print(state, flush=True)
        action = input("Your action: ")
        return self.from_str(action)

    def observe(self, reward: int, final_state: BaseGameState | None = None) -> None:
        if self.clear_outputs:
            clear_output()
        if reward == 1:
            print("You win!")
        elif reward == 0:
            print("It`s a draw.")
        else:
            print("You are lose :(")
        if final_state is not None:
            print("Final state:")
            print(final_state)
