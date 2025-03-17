from unittest.mock import Mock

from agent.player.random_ import RandomPlayer


def test_random_player_valid_action():
    player = RandomPlayer()
    mock_state = Mock()
    mock_state.actions = [(0, 0), (1, 1)]
    action = player.do_action(mock_state)
    assert action in mock_state.actions
