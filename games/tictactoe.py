from dataclasses import dataclass
from typing import ClassVar, Any

import numpy as np
from numpy.typing import NDArray

from games.base import BaseState, BaseGame, GameProcessError

type TicTacToeGrid = NDArray[np.int_]


@dataclass(eq=False)
class TicTacToeState(BaseState):
    grid: TicTacToeGrid
    _str_mapping: ClassVar[dict[int, str]] = {0: '_', 1: 'x', 2: 'o'}

    def __str__(self):
        return "\n".join('|'.join(self._str_mapping[i] for i in row) for row in self.grid)

    def __hash__(self):
        return hash(tuple(self.grid.flatten().tolist()))

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return np.all(self.grid == other.grid)

    @property
    def actions(self) -> list[int]:
        return np.argwhere(self.grid.flatten() == 0).flatten().tolist()

    @property
    def turn(self) -> int:
        # 1-based index of current player
        return self.turn0 + 1

    @property
    def turn0(self) -> int:
        # 0-based index of current player
        return self.grid.sum() % 3

    @property
    def is_terminal(self) -> bool:
        return bool(np.all(self.grid)) or self.is_win(1) or self.is_win(2)

    @property
    def winner(self) -> int | None:
        for player in [1, 2]:
            if self.is_win(player):
                return player
        return None

    def is_win(self, player: int) -> bool:
        if player not in [1, 2]:
            raise ValueError("'player' must be 1-based index of player")
        bool_grid = (self.grid == player)
        return bool(
            np.any(np.sum(bool_grid, axis=0) == 3) or
            np.any(np.sum(bool_grid, axis=1) == 3) or
            np.trace(bool_grid) == 3 or
            np.trace(np.fliplr(bool_grid)) == 3
        )

    def next(self, action: Any) -> Any:
        next_grid = self.grid.copy()
        row, col = action // 3, action % 3
        next_grid[row, col] = self.turn
        return TicTacToeState(next_grid)


class TicTacToe(BaseGame):
    def __init__(self) -> None:
        super().__init__()
        self._state: TicTacToeState = TicTacToeState(np.zeros((3, 3), np.int_))
        self._turn = 1
        self._is_terminated = False
        self._winner: int | None = -1

    def start(self) -> TicTacToeState:
        if self._state.grid.sum() != 0:
            raise GameProcessError("game already started")
        return self._state

    def step(self, action: int) -> TicTacToeState:
        if self._is_terminated:
            raise GameProcessError("game already finished")

        row, col = action // 3, action % 3
        if self._state.grid[row, col] != 0:
            raise ValueError("illegal action")

        self._state.grid[row, col] = self._turn
        self._is_terminated = self._state.is_terminal
        if self._is_terminated:
            if self._state.is_win(self._turn):
                self._winner = self._turn
            else:
                self._winner = None
        self._turn = 3 - self._turn
        return self._state

    def reset(self):
        self._state = TicTacToeState(np.zeros((3, 3), np.int_))
        self._turn = 1
        self._is_terminated = False

    @property
    def n_players(self) -> int:
        return 2

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def turn0(self) -> int:
        return self._turn - 1

    @property
    def is_terminated(self) -> bool:
        return self._is_terminated

    @property
    def winner(self) -> int | None:
        if self._winner == -1:
            raise AttributeError("game is ongoing")
        return self._winner


if __name__ == '__main__':
    grid_ = np.zeros((3, 3), dtype=int)
    grid_[1, 1] = 1
    grid_[0, 2] = 2
    start = TicTacToeState(grid_)
    print(start)
    print(dir(start))
    print(start.__dict__)
    print(id(start.__dict__))
    print(start.__slots__)
    print(start.__hash__)

    end = TicTacToeState(np.ones((3, 3), dtype=np.int_))
    print(end)
    print(end.__dict__)
    print(id(end.__dict__))
    print(end.__slots__)
    print(end.__hash__)
    # print(start == end)
    # print(hash(start))
    # print(TicTacToeState.grid)
    print(TicTacToeState.mro())
