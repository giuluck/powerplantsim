from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
from descriptors import classproperty

from ppsim import utils


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class DataType(ABC):
    """Abstract class that defines a datatype and has a unique key for comparison."""

    @dataclass
    class _InternalInfo:
        """Internal class to handle mutable (non-frozen) information about the simulation"""
        step: int = field(init=False, default=0)

    _info: _InternalInfo = field(init=False, default_factory=_InternalInfo)
    """Internal mutable simulation info."""

    @classproperty
    @abstractmethod
    def _properties(self) -> List[str]:
        """The list of public properties of the datatype."""
        pass

    @property
    @abstractmethod
    def key(self):
        """An identifier of the object."""
        pass

    @property
    @abstractmethod
    def _horizon(self) -> pd.Index:
        """The time horizon of the simulation in which the datatype is involved."""
        pass

    @property
    def _index(self):
        """The current index of the simulation as for the given time horizon."""
        return self._horizon[self._info.step]

    @property
    def dict(self) -> Dict[str, Any]:
        """A dictionary containing all the information of the datatype object indexed via property name."""
        return {param: getattr(self, param) for param in self._properties}

    def _step(self):
        """Checks and updates the internal simulation details.

        :return:
            The updated index of the simulation as for the given time horizon.
        """
        assert self._info.step < len(self._horizon), f"{self} has reached maximal number of updates for the simulation"
        index = self._index
        self._info.step += 1
        return index

    def _instance(self, other) -> bool:
        """Checks whether a different object is matching the self instance for comparison."""
        return isinstance(other, self.__class__)

    def __eq__(self, other: Any) -> bool:
        return self._instance(other) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({utils.stringify(value=self.key)})"
