from dataclasses import dataclass, field

from ppsim.datatypes.datatype import InternalDataType, DataType
from ppsim.datatypes.node import InternalNode, Node
from ppsim.utils import NamedTuple


@dataclass(repr=False, eq=False, slots=True)
class Edge(DataType):
    """An edge in the plant which can be exposed to the user."""

    source: Node = field(kw_only=True)
    """The source node."""

    destination: Node = field(kw_only=True)
    """The destination node."""

    min_flow: float = field(kw_only=True)
    """The minimal flow of commodity."""

    max_flow: float = field(kw_only=True)
    """The maximal flow of commodity."""

    integer: bool = field(kw_only=True)
    """Whether the flow must be integer or not."""

    @property
    def commodity(self) -> str:
        """The type of commodity that flows in the edge."""
        return self.destination.commodity_in

    def __repr__(self) -> str:
        return f"Edge(source='{self.source.name}', destination='{self.destination.name}')"


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class InternalEdge(InternalDataType):
    """An edge in the plant which is not exposed to the user."""

    @dataclass(frozen=True, unsafe_hash=True, slots=True)
    class EdgeID(NamedTuple):
        source: str = field()
        destination: str = field()

    source: InternalNode = field(kw_only=True)
    """The source node."""

    destination: InternalNode = field(kw_only=True)
    """The destination node."""

    min_flow: float = field(kw_only=True)
    """The minimal flow of commodity."""

    max_flow: float = field(kw_only=True)
    """The maximal flow of commodity."""

    integer: bool = field(kw_only=True)
    """Whether the flow must be integer or not."""

    def __post_init__(self):
        assert self.min_flow >= 0, f"The minimum flow cannot be negative, got {self.min_flow}"
        assert self.max_flow >= self.min_flow, \
            f"The maximum flow cannot be lower than the minimum, got {self.max_flow} < {self.min_flow}"
        assert self.destination.commodity_in is not None, \
            f"Destination node '{self.destination.name}' does not accept any input commodity, but it should"
        assert self.commodity in self.source.commodities_out, \
            f"Source node '{self.source.name}' should return commodity '{self.commodity}', " \
            f"but it returns {self.source.commodities_out}"

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
    def commodity(self) -> str:
        """The type of commodity that flows in the edge, which can be uniquely determined by the destination input."""
        return self.destination.commodity_in

    @property
    def key(self) -> EdgeID:
        return InternalEdge.EdgeID(source=self.source.name, destination=self.destination.name)

    @property
    def exposed(self) -> Edge:
        return Edge(
            source=self.source.exposed,
            destination=self.destination.exposed,
            min_flow=self.min_flow,
            max_flow=self.max_flow,
            integer=self.integer
        )
