from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set

from descriptors import classproperty

from ppsim.datatypes.datatype import DataType


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False)
class Node(DataType, ABC):
    """Data class for an abstract node in the plant."""

    name: str = field()
    """The name of the node."""

    @classproperty
    @abstractmethod
    def kind(self) -> str:
        """The node type."""
        pass

    @classproperty
    @abstractmethod
    def commodity_in(self) -> bool:
        """Whether the node accepts input commodities."""
        pass

    @classproperty
    @abstractmethod
    def commodity_out(self) -> bool:
        """Whether the node accepts output commodities."""
        pass

    @property
    @abstractmethod
    def commodities_in(self) -> Set[str]:
        """The set of input commodities that is accepted."""
        pass

    @property
    @abstractmethod
    def commodities_out(self) -> Set[str]:
        """The set of output commodities that is returned."""
        pass

    @property
    def key(self) -> str:
        return self.name

    def _instance(self, other) -> bool:
        return isinstance(other, Node)
