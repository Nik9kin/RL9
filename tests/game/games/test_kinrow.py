import pytest

import numpy as np

from game.games.kinrow import KInARowGameState


def test_rectangular_grid_equality():
    grid1 = np.array([[1, 0], [0, 0]])
    grid2 = np.array([[1, 0], [0, 0]])
    state1 = KInARowGameState(grid1, 2)
    state2 = KInARowGameState(grid2, 2)
    assert state1 == state2


def test_rectangular_grid_hash():
    grid = np.array([[1, 0], [0, 2]])
    state = KInARowGameState(grid, 2)
    expected_hash = hash((1, 0, 0, 2))
    assert hash(state) == expected_hash


def test_rectangular_grid_next_valid():
    grid = np.zeros((2, 3), dtype=int)
    state = KInARowGameState(grid, 2)
    state = state.next((0, 0))
    assert state.grid[0, 0] == 1  # Player 1's turn
    state = state.next((1, 2))
    assert state.grid[1, 2] == 2  # Player 2's turn


def test_rectangular_grid_next_invalid():
    grid = np.array([[1, 0], [0, 0]])
    state = KInARowGameState(grid, 2)
    with pytest.raises(ValueError):
        state.next((0, 0))


def test_rectangular_grid_actions():
    grid = np.array([[1, 0], [0, 2]])
    state = KInARowGameState(grid, 2)
    actions = state.actions
    assert [0, 1] in actions
    assert [1, 0] in actions
    assert len(actions) == 2


def test_rectangular_grid_is_terminal_filled():
    grid = np.array([[1, 2, 1], [1, 2, 2]])
    state = KInARowGameState(grid, 2)
    assert state.is_filled is True
    assert state.is_terminal is True


def test_turn_calculation():
    grid = np.zeros((2, 3), dtype=int)
    state = KInARowGameState(grid, 2)
    assert state.turn0 == 0
    assert state.turn == 1
    grid = np.array([[1, 0], [0, 0]])
    state = KInARowGameState(grid, 2)
    assert state.turn0 == 1
    assert state.turn == 2
    grid = np.array([[1, 2], [0, 0]])
    state = KInARowGameState(grid, 2)
    assert state.turn0 == 0
    assert state.turn == 1
