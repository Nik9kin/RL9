import numpy as np

from ..core.base import BaseGame, BasePlayer


class GameManager:
    def __init__(
            self,
            game: BaseGame,
            *players: BasePlayer,
            n_plays: int = 1,
    ):
        self.game = game
        self.players = players
        self.n_plays = n_plays

        self._winners = np.zeros(n_plays, np.int_)

    def run_single_game(self):
        state = self.game.start()
        while not self.game.is_terminated:
            action = self.players[self.game.turn0].do_action(state)
            state = self.game.step(action)

        for player, reward in zip(self.players, self.game.rewards):
            player.observe(reward, state)
        return self.game.winner

    def run(self):
        for i in range(self.n_plays):
            winner = self.run_single_game()
            self._winners[i] = winner if winner is not None else 0
            self.game.reset()
        return {
            "winners": self._winners,
        }
