from collections import namedtuple
from dataclasses import dataclass, field

from ppsim.datatypes.datatype import DataType
from ppsim.datatypes.node import Node


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False)
class Edge(DataType):
    """An edge in the plant."""

    EdgeID = namedtuple('EdgeID', 'source destination commodity')
    """Named tuple for edge identification."""

    source: Node = field(kw_only=True)
    """The source node."""

    destination: Node = field(kw_only=True)
    """The destination node."""

    commodity: str = field(kw_only=True)
    """The type of commodity that flows in the edge."""

    min_flow: float = field(kw_only=True)
    """The minimal flow of commodity."""

    max_flow: float = field(kw_only=True)
    """The maximal flow of commodity."""

    integer: bool = field(kw_only=True)
    """Whether the flow must be integer or not."""

    def __post_init__(self):
        assert self.min_flow >= 0, f"The minimum flow cannot be negative, got {self.min_flow}"
        assert self.max_flow >= self.min_flow, f"The maximum flow cannot be lower than the minimum, got {self.max_flow}"
        assert self.commodity in self.source.commodities_out, \
            f"Source node should return {self.commodity}, but it returns {self.source.commodities_out} only"
        assert self.commodity in self.destination.commodities_in, \
            f"Destination node should accept {self.commodity}, but it accepts {self.destination.commodities_in} only"

    def check(self, flow: float) -> bool:
        """Checks that the given flow falls within the range of the edge and respects optional integrality constraints.

        :param flow:
            The flow to check.

        :return:
            Whether the flow is valid or not.
        """
        # check bounds
        bounds = self.min_flow <= flow <= self.max_flow
        # check integrality (i.e., no integer constraint or flow is int/numpy integer)
        integrality = not self.integer or isinstance(flow, int) or flow.is_integer()
        # returns the combination of both checks
        return bounds and integrality

    @property
    def key(self) -> EdgeID:
        return Edge.EdgeID(source=self.source.key, destination=self.destination.key, commodity=self.commodity)
