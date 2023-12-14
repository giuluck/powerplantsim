from dataclasses import dataclass, field
from typing import Any, List

import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.datatype import DataType
from ppsim.datatypes.node import Node
from ppsim.utils import EdgeID
from ppsim.utils.typing import Flow


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Edge(DataType):
    """An edge in the plant."""

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

    _horizon: pd.Index = field(kw_only=True)
    """The time horizon of the simulation in which the datatype is involved."""

    _flows: pd.Series = field(init=False, default_factory=lambda: pd.Series(dtype=float))
    """The series of actual flows, which is filled during the simulation."""

    def __post_init__(self):
        assert self.min_flow >= 0, f"The minimum flow cannot be negative, got {self.min_flow}"
        assert self.max_flow >= self.min_flow, \
            f"The maximum flow cannot be lower than the minimum, got {self.max_flow} < {self.min_flow}"
        assert self.destination.commodity_in is not None, \
            f"Destination node '{self.destination.name}' does not accept any input commodity, but it should"
        assert self.commodity in self.source.commodities_out, \
            f"Source node '{self.source.name}' should return commodity '{self.commodity}', " \
            f"but it returns {self.source.commodities_out}"

    @classproperty
    def _properties(self) -> List[str]:
        return ['source', 'destination', 'commodity', 'min_flow', 'max_flow', 'integer', 'flows']

    @property
    def commodity(self) -> str:
        """The type of commodity that flows in the edge, which can be uniquely determined by the destination input."""
        return self.destination.commodity_in

    @property
    def flows(self) -> pd.Series:
        """The series of actual flows, which is filled during the simulation."""
        return self._flows.copy()

    @property
    def key(self) -> EdgeID:
        return self.source.name, self.destination.name

    def flow_at(self, index: Any) -> float:
        """Returns the flow at the given index.

        :param index:
            The time step of the simulation.

        :return:
            The corresponding flow.
        """
        return self._flows[index]

    def update(self, flow: Flow):
        """Updates the simulation by checking the value of the flow and appending it to the internal history.

        :param flow:
            The random number generator to be used for reproducible results.
        """
        index = self._step()
        assert flow >= self.min_flow, f"Flow for edge {self.key} should be >= {self.min_flow}, got {flow}"
        assert flow <= self.max_flow, f"Flow for edge {self.key} should be <= {self.max_flow}, got {flow}"
        assert not self.integer or flow.is_integer(), f"Flow for edge {self.key} should be integer, got {flow}"
        self._flows[index] = flow
