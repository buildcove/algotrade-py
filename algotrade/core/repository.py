from abc import ABC, abstractmethod

from algotrade.core import types


class BaseRepository(ABC):
    @abstractmethod
    def insert_position(self, position: types.Position):
        raise NotImplementedError

    @abstractmethod
    def insert_closed_position(self, closed_position: types.ClosedPosition):
        raise NotImplementedError

    @property
    @abstractmethod
    def positions(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def closed_positions(self):
        raise NotImplementedError


class Repository(BaseRepository):
    def __init__(self):
        self._positions = []
        self._closed_positions = []

    def insert_position(self, position: types.Position):
        self._positions.append(position)

    def insert_closed_position(self, closed_position: types.ClosedPosition):
        self._closed_positions.append(closed_position)

    @property
    def positions(self):
        return self._positions

    @property
    def closed_positions(self):
        return self._closed_positions

    @property
    def trades(self):
        return self.positions + self.closed_positions
