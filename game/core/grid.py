from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, ClassVar, Collection, Generator

import numpy as np
from numpy.typing import NDArray

from .base import BaseGameState
from .exception import IllegalActionError


@dataclass
class RectangularGridTwoPlayerState(BaseGameState, ABC):
    grid: NDArray[np.int_]
    _str_mapping: ClassVar[dict[int, str]] = {0: '_', 1: 'x', 2: 'o'}

    def __post_init__(self) -> None:
        if self.grid.ndim != 2:
            raise ValueError(f"grid must be two dimensional, got {self.grid.ndim} instead")

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

    def next(self, action: Collection[Integral]):
        if not isinstance(action, Collection) or len(action) != 2:
            raise IllegalActionError("unrecognized action format")
        if self.grid[*action] != 0:
            raise IllegalActionError("attempt to move into an occupied cell")
        next_grid = self.grid.copy()
        next_grid[*action] = self.turn
        return replace(self, grid=next_grid)

    @property
    def actions(self) -> Any:
        return np.argwhere(self.grid == 0).tolist()

    @property
    def is_filled(self) -> bool:
        return bool(np.all(self.grid))

    @property
    def is_terminal(self) -> bool:
        return self.is_filled or self.is_winner(1) or self.is_winner(2)

    @property
    def turn(self) -> int:
        # 1-based index of current player
        return self.turn0 + 1

    @property
    def turn0(self) -> int:
        # 0-based index of current player
        return self.grid.sum() % 3

    @property
    def winner(self) -> int | None:
        for player in [1, 2]:
            if self.is_winner(player):
                return player
        if self.is_filled:
            return None
        raise AttributeError("state is not terminal")
