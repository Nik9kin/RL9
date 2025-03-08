import numpy as np

from game.games.tictactoe import TicTacToeState, TicTacToe


def test_tictactoe_state_horizontal_win():
    grid = np.array([[1, 1, 1],
                     [2, 0, 2],
                     [0, 0, 0]])
    state = TicTacToeState(grid)
    assert state.is_winner(1)


def test_tictactoe_state_vertical_win():
    grid = np.array([[1, 2, 1],
                     [1, 2, 0],
                     [0, 2, 1]])
    state = TicTacToeState(grid)
    assert state.is_winner(2)


def test_tictactoe_state_diagonal_win():
    grid = np.array([[1, 2, 0],
                     [2, 1, 0],
                     [0, 2, 1]])
    state = TicTacToeState(grid)
    assert state.is_winner(1)
    grid = np.array([[1, 2, 2],
                     [1, 2, 0],
                     [2, 1, 1]])
    state = TicTacToeState(grid)
    assert state.is_winner(2)


def test_tictactoe_state_no_win():
    grid = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 2]])
    state = TicTacToeState(grid)
    assert not state.is_winner(1)
    assert not state.is_winner(2)


def test_tictactoe_next_action_conversion():
    state = TicTacToeState(np.zeros((3, 3), dtype=np.int_))
    next_state = state.next((1, 1))
    assert next_state.grid[1, 1] == 1


def test_tictactoe_game_start():
    game = TicTacToe()
    game.reset()
    state = game.start()
    assert np.all(state.grid == 0)


def test_tictactoe_game_step_win():
    game = TicTacToe()
    game.start()
    moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]  # Player 1 wins with first row
    for move in moves:
        game.step(move)
    assert game.is_terminated
    assert game.winner == 1


def test_tictactoe_game_draw():
    game = TicTacToe()
    game.start()
    # Moves leading to a draw
    # o|x|o
    # o|x|x
    # x|o|x
    moves = [(1, 1), (0, 0), (2, 2), (0, 2), (0, 1), (2, 1), (1, 2), (1, 0), (2, 0)]
    for move in moves:
        game.step(move)
    assert game.is_terminated
    assert game.winner is None
