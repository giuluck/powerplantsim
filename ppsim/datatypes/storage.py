from dataclasses import dataclass, field
from typing import Set, Callable, Any

from ppsim.datatypes.node import Node


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False)
class Storage(Node):
    """A node in the plant that stores certain commodities."""

    _capacity: Callable[[Any, str], float] = field(kw_only=True)
    """A function f(state, commodity) -> capacity which returns the maximal storing capacity of each commodity."""

    @property
    def commodities_in(self) -> Set[str]:
        raise NotImplementedError()

    @property
    def commodities_out(self) -> Set[str]:
        raise NotImplementedError()

    def capacity(self, state: Any, commodity: str) -> float:
        """Returns the storage capacity of a certain commodity in the given state.

        :param state:
            The state identifier.

        :param commodity:
            The commodity identifier.

        :return:
            The storage capacity of the machine for the given commodity in the given state.
        """
        return self._capacity(state, commodity)