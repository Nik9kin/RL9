import numpy as np
import pytest

from game.core.exception import IllegalActionError
from game.games.connectfour import ConnectFour, ConnectFourState


def test_next():
    grid0 = np.zeros((10, 10), dtype=int)
    grid1 = grid0.copy()
    grid1[9, 0] = 1
    grid2 = grid1.copy()
    grid2[9, 3] = 2
    grid3 = grid2.copy()
    grid3[8, 0] = 1
    state = ConnectFourState(grid0)
    with pytest.raises(IllegalActionError):
        state = state.next((0, 0))
    state = state.next((9, 0))
    assert np.all(state.grid == grid1)
    state = state.next(3)
    assert np.all(state.grid == grid2)
    next_state = state.next(0, inplace=True)
    assert state is next_state
    assert np.all(state.grid == grid3)


def test_vertical_win():
    grid = np.zeros((6, 7), dtype=int)
    for row in [5, 4, 3, 2]:
        grid[row, 3] = 1
    state = ConnectFourState(grid)
    assert state.is_winner(1)


def test_horizontal_win():
    grid = np.zeros((6, 7), dtype=int)
    grid[5, 1:5] = 2
    state = ConnectFourState(grid)
    assert state.is_winner(2)


def test_next_drop():
    state = ConnectFourState(np.zeros((6, 7), dtype=int))
    next_state = state.next(3)  # Column 3
    assert next_state.grid[5, 3] == 1


def test_actions_full_column():
    grid = np.ones((6, 1), dtype=int)  # Column 0 is full
    state = ConnectFourState(grid)
    assert 0 not in state.actions


def test_game_win():
    game = ConnectFour()
    game.start()
    # Player 1 drops in column 3 four times
    for _ in range(3):
        game.step(3)
        game.step(0)  # Player 2 in column 0
    game.step(3)
    assert game.is_terminated
    assert game.winner == 1
