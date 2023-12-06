from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Set

from ppsim.datatypes.datatype import DataType


class Nodes(Enum):
    """Enum class for node types."""
    SUPPLIER = 'supplier'
    CLIENT = 'client'
    MACHINE = 'machine'
    STORAGE = 'storage'

    def __repr__(self) -> str:
        return self.value.upper()


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False)
class Node(DataType, ABC):
    """Data class for an abstract node in the plant."""

    name: str = field()
    """The name of the node."""

    kind: Nodes = field(kw_only=True)
    """The node type."""

    commodity_in: bool = field(kw_only=True)
    """Whether the node accepts input commodities."""

    commodity_out: bool = field(kw_only=True)
    """Whether the node accepts output commodities."""

    @property
    def key(self) -> str:
        return self.name

    @property
    def type(self) -> str:
        """The node type."""
        return self.kind.value

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

    def _instance(self, other) -> bool:
        return isinstance(other, Node)
