from dataclasses import dataclass
from numbers import Integral
from typing import Collection

import numpy as np
from scipy.signal import convolve

from ..core.base import BaseGame
from ..core.exception import GameProcessError
from ..core.grid import RectangularGridTwoPlayerState


@dataclass(eq=False)
class KInARowGameState(RectangularGridTwoPlayerState):
    k: int

    def is_winner(self, player: int):
        if player not in [1, 2]:
            raise ValueError("'player' must be 1-based index of player")
        bool_grid = (self.grid == player)
        return np.any([
            np.any(convolve(bool_grid, kernel, mode="valid", method="direct") == self.k)
            for kernel in [
                np.ones((1, self.k)),
                np.ones((self.k, 1)),
                np.eye(self.k),
                np.fliplr(np.eye(self.k))
            ]
        ])


class KInARowGame(BaseGame):
    def __init__(self, rows: int, cols: int, k: int, *, state_class: type = KInARowGameState):
        super().__init__(state_class=state_class)
        self.rows = rows
        self.cols = cols
        self.k = k
        self._state: KInARowGameState = self._state_class(
            grid=np.zeros((rows, cols), dtype=np.int_),
            k=k,
        )
        self._n_players = 2
        self._is_terminated = False
        self._winner = None
        self._turn = 1

    def reset(self) -> None:
        self._state = self._state_class(
            grid=np.zeros((self.rows, self.cols), dtype=np.int_),
            k=self.k,
        )
        self._is_terminated = False
        self._winner = None
        self._turn = 1

    def start(self) -> KInARowGameState:
        if self._state.grid.sum() != 0:
            raise GameProcessError("game already started")
        return self._state

    def step(self, action: Collection[Integral]) -> KInARowGameState:
        if self._is_terminated:
            raise GameProcessError("game already finished")

        self._state = self._state.next(action)
        self._is_terminated = self._state.is_terminal
        if self._is_terminated and self._state.is_winner(self._turn):
            self._winner = self._turn
        self._turn = 3 - self._turn
        return self._state

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def turn0(self) -> int:
        return self._turn - 1
