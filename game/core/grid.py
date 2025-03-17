from __future__ import annotations
from abc import ABC
from collections.abc import Collection
from dataclasses import dataclass, replace, field
from typing import Any, ClassVar, Generator

import numpy as np
from numpy.typing import NDArray

from .base import BaseGameState
from .exception import IllegalActionError


type IntPair = tuple[int, int] | list[int]


@dataclass
class RectangularGridTwoPlayerState(BaseGameState, ABC):
    """
    Represents a game state for two-player grid-based games.

    Attributes:
        grid: 2D numpy array representing the game board.
    """

    grid: NDArray[np.int_]
    last_action: IntPair | None = field(default=None, kw_only=True)
    _str_mapping: ClassVar[dict[int, str]] = {0: '_', 1: 'x', 2: 'o'}

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return np.all(self.grid == other.grid)

    def __hash__(self) -> int:
        return hash(tuple(self.grid.flatten().tolist()))

    def __str__(self) -> str:
        return "\n".join(self.__list_str__())

    def __list_str__(self) -> Generator[str]:
        return ('|'.join(self._str_mapping[i] for i in row) for row in self.grid)

    def _next_no_val(self, action: IntPair, *, inplace: bool = False):
        if inplace:
            self.grid[*action] = self.turn
            self.last_action = action
            return self

        next_grid = self.grid.copy()
        next_grid[*action] = self.turn
        return replace(self, grid=next_grid, last_action=action)

    def _validate_action(self, action: IntPair) -> None:
        if not isinstance(action, Collection) or len(action) != 2:
            raise IllegalActionError("unrecognized action format")

        if self.grid[*action] != 0:
            raise IllegalActionError("attempt to move into an occupied cell")

    def next(self, action: IntPair, *, inplace: bool = False):
        """
        Create next state by placing current player's mark at action position.

        Args:
            action: (row, column) pair of ints indicating placement position

        Raises:
            IllegalActionError: If position is occupied or invalid
        """
        self._validate_action(action)
        return self._next_no_val(action, inplace=inplace)

    @property
    def actions(self) -> Any:
        """List of valid (row, column) positions as lists."""
        return np.argwhere(self.grid == 0).tolist()

    @property
    def is_filled(self):
        """Check if grid has no empty cells."""
        return np.all(self.grid)

    @property
    def is_terminal(self) -> bool:
        """Check if game has concluded (win or draw)."""
        return self.is_filled or self.is_winner(3 - self.turn)

    @property
    def turn(self) -> int:
        """Current player's turn (1-based index)."""
        if self.last_action is None:
            return 1 + self.grid.sum() % 3
        else:
            return 3 - self.grid[*self.last_action]

    @property
    def turn0(self) -> int:
        """Current player's turn (0-based index)."""
        return self.turn - 1

    @property
    def winner(self) -> int | None:
        """Get winning player or None for draw. Raises error if game ongoing."""
        prev_player = 3 - self.turn
        if self.is_winner(prev_player):
            return prev_player
        if self.is_filled:
            return None
        raise AttributeError("state is not terminal")
