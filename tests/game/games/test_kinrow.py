import pytest

import numpy as np

from game.core.exception import IllegalActionError
from game.games.kinrow import KInARowGameState, KInARowGame


def test_equality():
    grid1 = np.array([[1, 2], [0, 0]])
    grid2 = np.array([[1, 2], [0, 0]])
    state1 = KInARowGameState(grid1, 2)
    state2 = KInARowGameState(grid2, 2)
    assert state1 == state2
    state1 = KInARowGameState(grid1, 2, last_action=(0, 0))
    state2 = KInARowGameState(grid2, 2, last_action=(0, 1))
    assert state1 == state2


def test_hash():
    grid = np.array([[1, 0], [0, 2]])
    state = KInARowGameState(grid, 2)
    expected_hash = hash((1, 0, 0, 2))
    assert hash(state) == expected_hash


def test_next():
    grid0 = np.array([[0, 0], [0, 0]])
    grid1 = np.array([[1, 0], [0, 0]])
    grid2 = np.array([[1, 0], [0, 2]])
    state = KInARowGameState(grid0, 2)
    state = state.next((0, 0))
    assert np.all(state.grid == grid1)
    state = state.next((1, 1))
    assert np.all(state.grid == grid2)
    state = KInARowGameState(grid0, 2)
    next_state = state.next((0, 0), inplace=True)
    assert state is next_state


def test_next_invalid():
    grid = np.array([[1, 0], [0, 0]])
    state = KInARowGameState(grid, 2)
    with pytest.raises(IllegalActionError):
        state.next((0, 0))


def test_actions():
    grid = np.array([[1, 0], [0, 2]])
    state = KInARowGameState(grid, 2)
    actions = state.actions
    assert [0, 1] in actions
    assert [1, 0] in actions
    assert len(actions) == 2


def test_is_terminal_filled():
    grid = np.array([[1, 2, 1], [1, 2, 2]])
    state = KInARowGameState(grid, 2)
    assert state.is_filled
    assert state.is_terminal


def test_turn_calculation():
    grid = np.zeros((2, 3), dtype=int)
    state = KInARowGameState(grid, 2)
    assert state.turn0 == 0
    assert state.turn == 1
    state = state.next((1, 1))
    assert state.turn0 == 1
    assert state.turn == 2
    state.next((1, 0), inplace=True)
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

    grid = np.array([[1, 0], [0, 0]])
    state = KInARowGameState(grid, 2, last_action=(0, 0))
    assert state.turn0 == 1
    assert state.turn == 2

    grid = np.array([[1, 2], [0, 0]])
    state = KInARowGameState(grid, 2, last_action=(0, 1))
    assert state.turn0 == 0
    assert state.turn == 1

    grid = np.array([[1, 2], [0, 1]])
    state = KInARowGameState(grid, 2, last_action=(0, 0))
    assert state.turn0 == 1
    assert state.turn == 2

    grid = np.array([[1, 2], [0, 1]])
    state = KInARowGameState(grid, 2, last_action=(1, 1))
    assert state.turn0 == 1
    assert state.turn == 2


def test_state_winner():
    grid = np.array([[1, 2], [0, 1]])
    state = KInARowGameState(grid, 2)
    assert state.winner == 1
    assert state.is_winner(1)
    assert not state.is_winner(2)

    grid = np.array([[1, 2, 0], [0, 0, 1]])
    state = KInARowGameState(grid, 2)
    with pytest.raises(AttributeError):
        assert state.winner
    assert not state.is_winner(1)
    assert not state.is_winner(2)

    grid = np.array([[1, 2, 2], [0, 0, 1]])
    state = KInARowGameState(grid, 2, last_action=(0, 2))
    assert state.winner == 2
    assert not state.is_winner(1)
    assert state.is_winner(2)

    grid = np.array([[0, 0], [0, 0]])
    state = KInARowGameState(grid, 2)
    for _ in range(3):
        state.next(state.actions[0], inplace=True)
    assert state.winner == 1
    assert state.is_winner(1)
    assert not state.is_winner(2)

    grid = np.zeros((3, 3), dtype=int)
    state = KInARowGameState(grid, 3)
    for action in [
        (1, 1), (0, 0), (2, 2),
        (2, 0), (1, 0), (1, 2),
        (2, 1), (0, 1), (0, 2),
    ]:
        state = state.next(action)
    assert state.winner is None
    assert not state.is_winner(1)
    assert not state.is_winner(2)


def test_game_reset():
    game = KInARowGame(2, 2, 2)
    game.start()
    for action in [(0, 0), (0, 1)]:
        game.step(action)
    game.reset()
    grid = np.array([[0, 0], [0, 0]])
    state = KInARowGameState(grid, 2)
    assert game.state == state
    assert not game.is_terminated
    assert game.turn == 1

    game = KInARowGame(2, 2, 2)
    game.start()
    for action in [(0, 0), (0, 1), (1, 0)]:
        game.step(action)
    game.reset()
    grid = np.array([[0, 0], [0, 0]])
    state = KInARowGameState(grid, 2)
    assert game.state == state
    assert not game.is_terminated
    assert game.turn == 1


def test_game_start():
    game = KInARowGame(2, 2, 2)
    start_state = game.start()
    grid = np.array([[0, 0], [0, 0]])
    state = KInARowGameState(grid, 2)
    assert start_state == state
    assert start_state is not game.state
    assert start_state is not game._state


def test_game_step():
    game = KInARowGame(2, 2, 2)
    game.start()
    next_state = game.step((0, 1))
    grid = np.array([[0, 1], [0, 0]])
    state = KInARowGameState(grid, 2)
    assert next_state == state
    assert next_state is not game.state
    assert next_state is not game._state
    assert game.turn == 2
    assert next_state.last_action == (0, 1)


def test_game_is_terminated():
    game = KInARowGame(2, 2, 2)
    game.start()
    for action in [(0, 0), (0, 1), (1, 0)]:
        game.step(action)
    assert game.is_terminated


def test_game_n_players():
    game = KInARowGame(100, 1000, 3)
    assert game.n_players == 2


def test_game_turn():
    game = KInARowGame(2, 2, 2)
    game.start()
    turn = 1
    assert game.turn == turn
    assert game.turn0 == turn - 1
    for action in [(0, 0), (0, 1), (1, 0)]:
        game.step(action)
        turn = 3 - turn
        assert game.turn == turn
        assert game.turn0 == turn - 1

    game = KInARowGame(3, 3, 3)
    game.start()
    assert game.turn == 1
    assert game.turn0 == 0
    for action in [(0, 0), (0, 1), (1, 1)]:
        game.step(action)
    assert game.turn == 2
    assert game.turn0 == 1
    for action in [(0, 2), (2, 2)]:
        game.step(action)
    assert game.turn == 2
    assert game.turn0 == 1


def test_game_winner_rewards_0():
    game = KInARowGame(3, 3, 3)
    game.start()
    for action in [
        (1, 1), (0, 0), (2, 2),
        (2, 0), (1, 0), (1, 2),
        (2, 1), (0, 1), (0, 2),
    ]:
        game.step(action)
    assert game.winner is None
    assert game.rewards == [0, 0]


def test_game_winner_rewards_1():
    game = KInARowGame(3, 3, 3)
    game.start()
    for action in [(0, 0), (0, 1), (1, 1)]:
        game.step(action)
    with pytest.raises(AttributeError):
        assert game.rewards
    with pytest.raises(AttributeError):
        assert game.winner
    for action in [(0, 2), (2, 2)]:
        game.step(action)
    assert game.rewards == [1, -1]
    assert game.winner == 1


def test_game_winner_rewards_2():
    game = KInARowGame(3, 3, 3)
    game.start()
    for action in [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 1)]:
        game.step(action)
    assert game.rewards == [-1, 1]
    assert game.winner == 2


def test_game_state_copy():
    game = KInARowGame(3, 3, 3)
    game.start()
    game.step((1, 1))
    assert game.state is not game._state
