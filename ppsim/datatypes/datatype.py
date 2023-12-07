from abc import abstractmethod, ABC
from dataclasses import field
from typing import Any

from ppsim.utils.strings import stringify


class DataType(ABC):
    """Abstract class that defines a datatype which has a unique key used for comparison."""

    plant: Any = field(kw_only=True, default=None)
    """The plant that contains the datatype or None if the datatype is not linked to any plant."""

    @property
    @abstractmethod
    def key(self) -> Any:
        """An identifier of the object."""
        pass

    def _instance(self, other) -> bool:
        """Checks whether a different object is matching the self instance for comparison."""
        return isinstance(other, self.__class__)

    def __eq__(self, other) -> bool:
        return self._instance(other) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return stringify(value=self.key)
