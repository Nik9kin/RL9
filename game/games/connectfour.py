from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..core.exception import IllegalActionError
from ..core.grid import IntPair
from .kinrow import KInARowGameState, KInARowGame


@dataclass(eq=False)
class ConnectFourState(KInARowGameState):
    """Connect Four game state with column-based piece placement and 4-in-a-row win condition."""

    k: int = 4

    def __str__(self) -> str:
        rows, cols = self.grid.shape
        if cols >= 100:
            return super().__str__()

        if cols <= 10:
            def col_label_to_str(n: int):
                return f'{n:<2d}'
            n_col_ticks = cols
            arr_sign = 'v '
        else:
            def col_label_to_str(n: int):
                return '  ' if n % 2 else f'{n:<2d}'
            n_col_ticks = (cols + 1) // 2
            arr_sign = 'v   '

        col_ticks = list(map(col_label_to_str, range(cols)))

        first_row = ' ' + ''.join(col_ticks) + '\n'
        second_row = ' ' + arr_sign * n_col_ticks + '\n'

        str_mapping = {0: ' ', 1: 'x', 2: 'o'}
        str_mapping_last_row = {0: '_', 1: 'x', 2: 'o'}
        return (
            first_row +
            second_row +
            '\n'.join(self._join('|', (str_mapping[i] for i in row)) for row in self.grid[:-1]) +
            '\n' + self._join('|', (str_mapping_last_row[i] for i in self.grid[-1])) + '\n'
        )

    @staticmethod
    def _join(sep: str, iterable: Iterable[str]):
        return sep + sep.join(iterable) + sep

    def _validate_action(self, action: int | IntPair) -> None:
        if isinstance(action, int):
            if self.grid[0, action] != 0:
                raise IllegalActionError("attempt to make a move into a filled column")
        else:
            super()._validate_action(action)
            row, col = action
            if row + 1 < self.grid.shape[0] and self.grid[row + 1, col] == 0:
                raise IllegalActionError("move can only be made to the bottom of the column")

    def next(self, action: int | IntPair, *, inplace: bool = False):
        """
        Handle column-based piece placement.

        Args:
            action: Column index or (row, column) tuple

        Returns:
            New state with piece placed in the lowest available row
        """
        self._validate_action(action)

        if isinstance(action, int):
            col = action
            row = np.sum(self.grid[:, col] == 0) - 1
            action = (row, col)

        return self._next_no_val(action, inplace=inplace)

    @property
    def actions(self) -> list[int]:
        """List of valid column indices with available space."""
        return np.argwhere(self.grid[0] == 0).flatten().tolist()


class ConnectFour(KInARowGame):
    """Connect Four game implementation with 6x7 grid by default."""

    def __init__(self, rows: int = 6, cols: int = 7, *, state_class: type = ConnectFourState):
        super().__init__(rows, cols, 4, state_class=state_class)
