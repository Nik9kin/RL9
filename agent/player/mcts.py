from __future__ import annotations

import copy
import logging
import time
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from agent.player import RandomPlayer
from game.core.base import BaseGameState, BasePlayer

from game import ConnectFour
from game.manager import GameManager
from agent.common import select_random_optimal_action
from agent.player import IOPlayer


class MCTSNode:
    def __init__(
            self,
            state: BaseGameState,
            *,
            parent: MCTSNode | None = None,
            ucb_c=1.41,
    ):
        self.state = state
        self.parent = parent

        self.actions = self.state.actions
        self.n_actions = len(self.actions)
        self._ucb_c = ucb_c

        self._value_estimates = np.zeros(self.n_actions, dtype=np.float64)
        self._action_attempts = np.zeros(self.n_actions, dtype=np.int64)
        self._last_action: int = -1
        self._t = 0

        self.children: list[MCTSNode] = []

        self._is_value_inexact = np.ones(self.n_actions, dtype=np.bool)
        self.n_inexact_estimates = self.n_actions

    def expand(self) -> None:
        self.children = [
            MCTSNode(self.state.next(action), parent=self, ucb_c=self._ucb_c)
            for action in self.actions
        ]

    def do_action(self) -> int:
        if self._t < self.n_actions:
            # Exploring start phase: cycle through actions
            self._last_action = self._t % self.n_actions
        else:
            # UCB1 selection: value + confidence bound
            exploration_bonus = self._ucb_c * np.sqrt(np.log(self._t) / self._action_attempts)
            priorities = self._value_estimates + exploration_bonus
            inexact_actions = np.nonzero(self._is_value_inexact)[0]
            self._last_action = inexact_actions[np.argmax(priorities[self._is_value_inexact])]

        self._action_attempts[self._last_action] += 1
        self._t += 1
        return self._last_action

    def observe(self, reward: int, exact: bool) -> tuple[int, bool]:
        a = self._last_action
        if exact:
            self._value_estimates[a] = reward
            self._is_value_inexact[a] = False
            self.n_inexact_estimates -= 1
            if reward == 1:
                return -1, True
            elif self.n_inexact_estimates == 0:
                return -self._value_estimates.max(), True
            else:
                return -reward, False
        else:
            step_size = 1 / self._action_attempts[a]
            self._value_estimates[a] += step_size * (reward - self._value_estimates[a])
            return -reward, False

    def just_select(self):
        return select_random_optimal_action(self._value_estimates, list(range(self.n_actions)))

    @property
    def action_attempts(self):
        return self._action_attempts

    @property
    def value_estimates(self):
        """np.ndarray: Current estimates of the value of each action."""
        return self._value_estimates

    @property
    def is_value_exact(self):
        return ~self._is_value_inexact


class MCTS(BasePlayer):
    def __init__(
            self,
            start_state: BaseGameState,
            *,
            time_for_action: float = 1.0,
            rollout_policy: str | BasePlayer = 'random',
            ucb_c: float = 1.41,
            logg: bool = False,
            log_file: str = 'mcts.log'
    ):
        self.start_state = copy.deepcopy(start_state)
        self.time_for_action = int(time_for_action * 10 ** 9)

        self.rollout_policy: BasePlayer
        if isinstance(rollout_policy, str):
            if rollout_policy == 'random':
                self.rollout_policy = RandomPlayer()
            else:
                raise ValueError(f'Unknown rollout_policy: "{rollout_policy}"')
        else:
            self.rollout_policy = rollout_policy

        self.ucb_c = ucb_c

        self.logger: logging.Logger | None
        if logg:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(
                filename=log_file,
                filemode='w',
                format='%(message)s',
                level=logging.INFO
            )
        else:
            self.logger = None

        self.root = MCTSNode(start_state, ucb_c=ucb_c)

    def __str__(self):
        return f"MCTS(time for action : {self.time_for_action / 10 ** 9:.3f} s)"

    def do_action(self, state: BaseGameState) -> Any:
        start_time = time.time_ns()

        if state != self.root.state:
            self.root = self._find_node(state)
            self.root.parent = None

        if not self.root.children:
            self.root.expand()

        while self.root.n_inexact_estimates and time.time_ns() - start_time < self.time_for_action:
            leaf = self._select_leaf()
            if not leaf.state.is_terminal:
                leaf.expand()
                leaf = leaf.children[leaf.do_action()]
            if leaf.state.is_terminal:
                winner = leaf.state.winner
                if winner is None:
                    reward = 0
                else:
                    reward = 2 * int(winner == leaf.state.turn) - 1
                exact = True
            else:
                reward = self._rollout(leaf)
                exact = False
            self._backprop(leaf, reward, exact)

        action_ind = self.root.just_select()
        action = self.root.actions[action_ind]
        if self.logger:
            self._log_info()
            self.logger.info(f"Selected action: {action}")
        self.root = self.root.children[action_ind]
        self.root.parent = None
        if self.logger:
            self._log_info()
            self.logger.info("Opponent turn")
        return action

    def reset(self):
        self.root = MCTSNode(self.start_state, ucb_c=self.ucb_c)

    def _select_leaf(self) -> MCTSNode:
        node = self.root
        while node.children:
            node = node.children[node.do_action()]
        return node

    def _rollout(self, node: MCTSNode) -> int:
        state = deepcopy(node.state)
        start_state_player = state.turn

        while not state.is_terminal:
            state.next(self.rollout_policy.do_action(state), inplace=True)

        winner = state.winner
        if winner is None:
            return 0
        else:
            return 2 * int(winner == start_state_player) - 1

    @staticmethod
    def _backprop(node: MCTSNode, reward: int, exact: bool):
        reward = -reward
        while node.parent is not None:
            node = node.parent
            reward, exact = node.observe(reward, exact)

    def _find_node(self, state: BaseGameState) -> MCTSNode:
        if self.root.state == state:
            return self.root

        if not self.root.children:
            self.root.expand()

        for child in self.root.children:
            if child.state == state:
                return child

        raise ValueError(f'State {state} is not reachable by the opponent')

    def _log_info(self):
        self.logger.info("")
        self.logger.info(f"Current state:\n{self.root.state}")
        actions_info = pd.DataFrame(
            {
                "Action": self.root.actions,
                "Value estimate": self.root.value_estimates,
                "Attempts": self.root.action_attempts,
                "Is exact": self.root.is_value_exact,
            }
        )
        self.logger.info(actions_info.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        ))


if __name__ == '__main__':
    connect_four = ConnectFour()
    start = connect_four.start()
    manager = GameManager(connect_four, IOPlayer('int'), MCTS(start, logg=True))
    # manager = GameManager(
    #     connect_four,
    #     MCTS(start, time_for_action=5.0, logg=True),
    #     IOPlayer('int')
    # )
    # manager = GameManager(
    #     connect_four,
    #     IOPlayer('int'),
    #     MCTS(start, time_for_action=5.0, logg=True)
    # )
    manager.run_single_game()
