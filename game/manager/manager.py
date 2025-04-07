import numpy as np
import pandas as pd
from tqdm.notebook import trange

from ..core.base import BaseGame, BasePlayer


class GameManager:
    """
    Manages multiple game sessions between players with result tracking.

    Attributes:
        game: Game instance to manage
        players: Sequence of players
        n_plays: Number of games to play
        shuffling: Order shuffling strategy ('no', 'circular', or 'random')
    """

    def __init__(
            self,
            game: BaseGame,
            *players: BasePlayer,
            n_plays: int = 1,
            shuffling: str = 'no',
            player_labels: list[str] | None = None,
    ):
        self.game = game
        self.players = players
        self.n_plays = n_plays
        if shuffling not in ['no', 'circular', 'random']:
            raise ValueError('"shuffling" must be one of "no", "circular" or "random"')
        self.shuffling = shuffling
        if player_labels is None:
            player_labels = list(map(str, players))
        self.player_labels = player_labels
        self.roles_labels = game.roles_descriptions

        self._winners_players = np.zeros(n_plays, dtype=np.int_)
        self._winners_roles = np.zeros(n_plays, dtype=np.int_)
        self._winners_matrix = np.zeros((len(players) + 1, len(players) + 1), dtype=np.int_)
        self._order = np.arange(len(players), dtype=np.int_)

    def run_single_game(self) -> int | None:
        """Execute one full game and return winner (None for draw)."""

        state = self.game.start()
        while not self.game.is_terminated:
            action = self.players[self._order[self.game.turn0]].do_action(state)
            state = self.game.step(action)

        for player_ind, reward in zip(self._order, self.game.rewards):
            self.players[player_ind].observe(reward, state)
        return self.game.winner

    def run(self, verbose: bool = False):
        """
        Run all games and collect statistics.

        Returns:
            Dictionary with:
            - 'winners (players)': array of winning player indices
            - 'winners (roles)': array of winning role indices
            - 'winners matrix': matrix counting role-player win combinations
        """

        iter_ = trange if verbose else range
        for i in iter_(self.n_plays):
            winner_role = self.run_single_game()
            if winner_role is None:
                self._winners_players[i] = 0
                self._winners_roles[i] = 0
                self._winners_matrix[0, 0] += 1
            else:
                winner_player = self._order[winner_role - 1] + 1
                self._winners_players[i] = winner_player
                self._winners_roles[i] = winner_role
                self._winners_matrix[winner_role, winner_player] += 1
            self.game.reset()
            for player in self.players:
                player.reset()
            if self.shuffling == 'circular':
                self._order = np.hstack((self._order[1:], self._order[:1]))
            elif self.shuffling == 'random':
                np.random.shuffle(self._order)

        return {
            "winners (players)": self._winners_players,
            "winners (roles)": self._winners_roles,
            "winners matrix": pd.DataFrame(
                self._winners_matrix,
                ["Draw"] + self.roles_labels,
                ["Draw"] + self.player_labels,
            ),
        }
