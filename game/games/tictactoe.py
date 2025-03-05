from dataclasses import dataclass

from .kinrow import KInARowGame, KInARowGameState


@dataclass(eq=False)
class TicTacToeState(KInARowGameState):
    k: int = 3


class TicTacToe(KInARowGame):
    def __init__(self, *, state_class: type = TicTacToeState):
        super().__init__(3, 3, 3, state_class=state_class)
