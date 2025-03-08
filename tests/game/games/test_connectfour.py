import numpy as np

from game.games.connectfour import ConnectFour, ConnectFourState


def test_connectfour_vertical_win():
    grid = np.zeros((6, 7), dtype=int)
    for row in [5, 4, 3, 2]:
        grid[row, 3] = 1
    state = ConnectFourState(grid)
    assert state.is_winner(1)


def test_connectfour_horizontal_win():
    grid = np.zeros((6, 7), dtype=int)
    grid[5, 1:5] = 2
    state = ConnectFourState(grid)
    assert state.is_winner(2)


def test_connectfour_next_drop():
    state = ConnectFourState(np.zeros((6, 7), dtype=int))
    next_state = state.next(3)  # Column 3
    assert next_state.grid[5, 3] == 1


def test_connectfour_actions_full_column():
    grid = np.ones((6, 1), dtype=int)  # Column 0 is full
    state = ConnectFourState(grid)
    assert 0 not in state.actions


def test_connectfour_game_win():
    game = ConnectFour()
    game.start()
    # Player 1 drops in column 3 four times
    for _ in range(3):
        game.step(3)
        game.step(0)  # Player 2 in column 0
    game.step(3)
    assert game.is_terminated
    assert game.winner == 1
