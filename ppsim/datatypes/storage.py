from dataclasses import dataclass, field
from typing import Set

from descriptors import classproperty

from ppsim.datatypes.node import InternalNode, Node


@dataclass()
class Storage(Node):
    """A node in the plant that stores certain commodities and can be exposed to the user."""

    capacity: float = field(kw_only=True)
    """The storage capacity, which must be a strictly positive number."""

    dissipation: float = field(kw_only=True)
    """The dissipation rate of the storage at every time unit, which must be a float in [0, 1]."""


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False)
class InternalStorage(InternalNode):
    """A node in the plant that stores certain commodities and is not exposed to the user."""

    commodity: str = field(kw_only=True)
    """The commodity stored by the machine."""

    capacity: float = field(kw_only=True)
    """The storage capacity, which must be a strictly positive number."""

    dissipation: float = field(kw_only=True)
    """The dissipation rate of the storage at every time unit, which must be a float in [0, 1]."""

    def __post_init__(self):
        assert self.capacity > 0.0, f"Capacity should be strictly positive, got {self.capacity}"
        assert 0.0 <= self.dissipation <= 1.0, f"Dissipation should be in range [0, 1], got {self.dissipation}"

    @classproperty
    def kind(self) -> str:
        return 'storage'

    @classproperty
    def commodity_in(self) -> bool:
        return True

    @classproperty
    def commodity_out(self) -> bool:
        return True

    @property
    def commodities_in(self) -> Set[str]:
        return {self.commodity}

    @property
    def commodities_out(self) -> Set[str]:
        return {self.commodity}

    @property
    def exposed(self) -> Storage:
        return Storage(
            name=self.name,
            commodities_in=self.commodities_in,
            commodities_out=self.commodities_out,
            capacity=self.capacity,
            dissipation=self.dissipation
        )
