from abc import ABC, abstractmethod
from typing import Any


class BaseState(ABC):
    @abstractmethod
    def __hash__(self) -> int:
        return 0

    @abstractmethod
    def __eq__(self, other) -> bool:
        return self is other

    @property
    @abstractmethod
    def actions(self) -> list[Any]: ...

    @property
    @abstractmethod
    def turn(self) -> int: ...

    @property
    @abstractmethod
    def turn0(self) -> int: ...

    @property
    @abstractmethod
    def is_terminal(self) -> bool: ...

    @property
    @abstractmethod
    def winner(self) -> int | None: ...

    @abstractmethod
    def next(self, action: Any) -> Any: ...


class BaseGame(ABC):
    def __init__(self) -> None:
        self._state: BaseState

    @abstractmethod
    def start(self) -> BaseState: ...  # raise error if game already started

    @abstractmethod
    def step(self, action: Any) -> BaseState: ...

    @abstractmethod
    def reset(self) -> None: ...

    @property
    def state(self) -> BaseState:
        return self._state

    @property
    @abstractmethod
    def n_players(self) -> int: ...

    @property
    def turn(self) -> int:
        return self._state.turn

    @property
    def turn0(self) -> int:
        return self._state.turn0

    @property
    def is_terminated(self) -> bool:
        return self._state.is_terminal

    @property
    @abstractmethod
    def winner(self) -> int | None: ...  # raise exception if game is not terminated


class BasePlayer(ABC):
    @abstractmethod
    def do_action(self, state: BaseState) -> Any: ...

    def observe(self, reward: int, final_state: BaseState | None = None) -> None:
        pass


class GameProcessError(RuntimeError):
    pass
