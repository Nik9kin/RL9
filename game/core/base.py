from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseGameState(ABC):
    """Abstract base class for game states."""

    @abstractmethod
    def __eq__(self, other) -> bool:
        return self is other

    @abstractmethod
    def __hash__(self) -> int:
        return 0

    @abstractmethod
    def is_winner(self, player: int) -> bool:
        """Check if player has won the game."""
        ...

    @abstractmethod
    def next(self, action: Any) -> BaseGameState:
        """Generate next state after applying action."""
        ...

    @property
    @abstractmethod
    def actions(self) -> list[Any]:
        """List of valid actions for current state."""
        ...

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if state is terminal (game ended)."""
        ...

    @property
    @abstractmethod
    def turn(self) -> int:
        """Current player's turn (1-based)."""
        ...

    @property
    @abstractmethod
    def turn0(self) -> int:
        """Current player's turn (0-based)."""
        ...

    @property
    @abstractmethod
    def winner(self) -> int | None:
        """Get winner if terminal, else raise exception."""
        ...


class BaseGame(ABC):
    """Abstract base class for game."""

    def __init__(self, *, state_class: type = BaseGameState) -> None:
        self._state_class: type = state_class
        self._state: BaseGameState
        self._n_players: int
        self._is_terminated: bool
        self._winner: int | None

    @abstractmethod
    def reset(self) -> None:
        """Reset game to initial state."""
        ...

    @abstractmethod
    def start(self) -> BaseGameState:
        """Start game and return initial state. Raise exception if game already started."""
        ...

    @abstractmethod
    def step(self, action: Any) -> BaseGameState:
        """Advance game by applying action."""
        ...

    @property
    def is_terminated(self) -> bool:
        """Check if game has concluded."""
        return self._is_terminated

    @property
    def n_players(self) -> int:
        """Number of players in the game."""
        return self._n_players

    @property
    def rewards(self) -> tuple[int, ...]:
        """Player rewards tuple. Only valid when terminated."""

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
        """Current game state."""
        return self._state

    @property
    def turn(self) -> int:
        """Current player's turn (1-based)."""
        return self._state.turn

    @property
    def turn0(self) -> int:
        """Current player's turn (0-based)."""
        return self._state.turn0

    @property
    def winner(self) -> int | None:
        """Get winner if game ended, else raise exception."""

        if not self._is_terminated:
            raise AttributeError("game is ongoing")
        return self._winner


class BasePlayer(ABC):
    """Abstract base class for game players."""

    @abstractmethod
    def do_action(self, state: BaseGameState) -> Any:
        """Choose action given current state."""
        ...

    def observe(self, reward: int, final_state: BaseGameState | None = None) -> None:
        """Receive game outcome observation."""
        pass
