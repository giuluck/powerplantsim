from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any

from ppsim import utils


@dataclass()
class DataType(ABC):
    """Abstract class that defines a datatype which can be exposed to the user."""
    pass


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class InternalDataType(ABC):
    """Abstract class that defines a datatype which is not exposed to the user and has a unique key for comparison."""

    @property
    @abstractmethod
    def key(self) -> Any:
        """An identifier of the object."""
        pass

    @property
    @abstractmethod
    def exposed(self) -> DataType:
        """The public datatype that can be exposed to the user."""
        pass

    def _instance(self, other) -> bool:
        """Checks whether a different object is matching the self instance for comparison."""
        return isinstance(other, self.__class__)

    def __eq__(self, other) -> bool:
        return self._instance(other) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return utils.stringify(value=self.key)
