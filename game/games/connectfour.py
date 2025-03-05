from dataclasses import dataclass
from numbers import Integral
from typing import Collection

import numpy as np

from ..core.exception import IllegalActionError
from .kinrow import KInARowGameState, KInARowGame


@dataclass(eq=False)
class ConnectFourState(KInARowGameState):
    k: int = 4

    def next(self, action: Integral | Collection[Integral]):
        if isinstance(action, Integral):
            col = action
            if self.grid[0, col] != 0:
                raise IllegalActionError("attempt to make a move into a filled column")
            row = np.sum(self.grid[:, col] == 0) - 1
            action = (row, col)
        return super().next(action)

    @property
    def actions(self) -> list[int]:
        return np.argwhere(self.grid[0] == 0).flatten().tolist()


class ConnectFour(KInARowGame):
    def __init__(self, rows: int = 6, cols: int = 7, *, state_class: type = ConnectFourState):
        super().__init__(rows, cols, 4, state_class=state_class)
