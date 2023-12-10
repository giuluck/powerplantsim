from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set, Optional

from descriptors import classproperty

from ppsim.datatypes.datatype import InternalDataType, DataType


@dataclass()
class Node(DataType):
    """Public data class for an abstract node in the plant which can be exposed to the user."""

    name: str = field(kw_only=True)
    """The name of the node."""

    commodity_in: Optional[str] = field(kw_only=True)
    """The (optional) input commodities that is accepted (can be at most one)."""

    commodities_out: Set[str] = field(kw_only=True)
    """The set of output commodities that is returned (can be more than one)."""


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False)
class InternalNode(InternalDataType, ABC):
    """Data class for an abstract node in the plant which is not exposed to the user."""

    name: str = field(kw_only=True)
    """The name of the node."""

    @classproperty
    @abstractmethod
    def kind(self) -> str:
        """The node type."""
        pass

    @property
    @abstractmethod
    def commodity_in(self) -> Optional[str]:
        """The (optional) input commodities that is accepted (can be at most one)."""
        pass

    @property
    @abstractmethod
    def commodities_out(self) -> Set[str]:
        """The set of output commodities that is returned (can be more than one)."""
        pass

    @property
    @abstractmethod
    def exposed(self) -> Node:
        pass

    @property
    def key(self) -> str:
        return self.name

    def _instance(self, other) -> bool:
        return isinstance(other, InternalNode)
