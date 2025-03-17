from __future__ import annotations

import copy
import time
from copy import deepcopy
from typing import Any

import numpy as np

from agent.player import RandomPlayer
from game.core.base import BaseGameState, BasePlayer
from bandits.agent import UCB1

from game import ConnectFour
from game.manager import GameManager
from agent.player import IOPlayer


class MCTSNode(UCB1):
    def __init__(self, state: BaseGameState, *, parent: MCTSNode | None = None, ucb_c=1.41):
        self.state = state
        self.parent = parent

        self.actions = self.state.actions

        super().__init__(len(self.actions), c=ucb_c)

        self.children: list[MCTSNode] = []

    def expand(self) -> None:
        self.children = [
            MCTSNode(self.state.next(action), parent=self, ucb_c=self.c)
            for action in self.actions
        ]

    def just_select(self):
        return np.argmax(self._value_estimates)


class MCTS(BasePlayer):
    def __init__(
            self,
            start_state: BaseGameState,
            *,
            time_for_action: float = 1.0,
            rollout_policy: str | BasePlayer = 'random',
            ucb_c: float = 1.41,
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

        self.root = MCTSNode(start_state, ucb_c=ucb_c)

    def do_action(self, state: BaseGameState) -> Any:
        start_time = time.time_ns()

        if state != self.root.state:
            self.root = self._find_node(state)
            self.root.parent = None

        if not self.root.children:
            self.root.expand()

        while time.time_ns() - start_time < self.time_for_action:
            leaf = self._select_leaf()
            if not leaf.state.is_terminal:
                leaf.expand()
                leaf = leaf.children[leaf.do_action()]
            reward = self._rollout(leaf)
            self._backprop(leaf, reward)

        action_ind = self.root.just_select()
        action = self.root.actions[action_ind]
        self.root = self.root.children[action_ind]
        self.root.parent = None
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
    def _backprop(node: MCTSNode, reward: int):
        while node.parent is not None:
            node = node.parent
            reward = -reward
            node.observe(reward)

    def _find_node(self, state: BaseGameState) -> MCTSNode:
        if self.root.state == state:
            return self.root

        if not self.root.children:
            self.root.expand()

        for child in self.root.children:
            if child.state == state:
                return child

        raise ValueError(f'State {state} is not reachable by the opponent')


if __name__ == '__main__':
    connect_four = ConnectFour()
    start = connect_four.start()
    manager = GameManager(connect_four, IOPlayer('int'), MCTS(start))
    manager.run_single_game()
