from dataclasses import dataclass

from .kinrow import KInARowGame, KInARowGameState


@dataclass(eq=False)
class TicTacToeState(KInARowGameState):
    """Tic-Tac-Toe game state with 3x3 grid and 3-in-a-row win condition."""
    k: int = 3


class TicTacToe(KInARowGame):
    """Tic-Tac-Toe game implementation."""
    def __init__(self, *, state_class: type = TicTacToeState):
        super().__init__(3, 3, 3, state_class=state_class)
