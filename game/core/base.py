from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseGameState(ABC):
    @abstractmethod
    def __eq__(self, other) -> bool:
        return self is other

    @abstractmethod
    def __hash__(self) -> int:
        return 0

    @abstractmethod
    def is_winner(self, player: int) -> bool: ...

    @abstractmethod
    def next(self, action: Any) -> BaseGameState: ...

    @property
    @abstractmethod
    def actions(self) -> list[Any]: ...

    @property
    @abstractmethod
    def is_terminal(self) -> bool: ...

    @property
    @abstractmethod
    def turn(self) -> int: ...

    @property
    @abstractmethod
    def turn0(self) -> int: ...

    @property
    @abstractmethod
    def winner(self) -> int | None: ...  # raise exception if game is not terminated


class BaseGame(ABC):
    def __init__(self, *, state_class: type = BaseGameState) -> None:
        self._state_class: type = state_class
        self._state: BaseGameState
        self._n_players: int
        self._is_terminated: bool
        self._winner: int | None

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def start(self) -> BaseGameState: ...  # raise exception if game already started

    @abstractmethod
    def step(self, action: Any) -> BaseGameState: ...

    @property
    def is_terminated(self) -> bool:
        return self._is_terminated

    @property
    def n_players(self) -> int:
        return self._n_players

    @property
    def rewards(self) -> tuple[int, ...]:
        if not self._is_terminated:
            raise AttributeError("game is ongoing")
        if self._winner is None:
            return (0,) * self._n_players
        else:
            rewards = [-1] * self._n_players
            rewards[self._winner - 1] = 1
            return tuple(rewards)

    @property
    def state(self) -> BaseGameState:
        return self._state

    @property
    def turn(self) -> int:
        return self._state.turn

    @property
    def turn0(self) -> int:
        return self._state.turn0

    @property
    def winner(self) -> int | None:
        if not self._is_terminated:
            raise AttributeError("game is ongoing")
        return self._winner


class BasePlayer(ABC):
    @abstractmethod
    def do_action(self, state: BaseGameState) -> Any: ...

    def observe(self, reward: int, final_state: BaseGameState | None = None) -> None:
        pass
