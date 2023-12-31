from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
# noinspection PyPackageRequirements
from descriptors import classproperty

from ppsim.datatypes.datatype import DataType
from ppsim.datatypes.node import Node
from ppsim.utils.typing import SingleEdgeID, Flow, MultiEdgeID, State


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class Edge(DataType, ABC):
    """An (abstract) edge in the plant."""

    _source: Node = field(kw_only=True)
    """The source node."""

    _destination: Node = field(kw_only=True)
    """The destination node."""

    commodity: str = field(kw_only=True)
    """The type of commodity that flows in the edge, which can be uniquely determined by the destination input."""

    min_flow: float = field(kw_only=True)
    """The minimal flow of commodity."""

    max_flow: float = field(kw_only=True)
    """The maximal flow of commodity."""

    _flows: List[float] = field(init=False, default_factory=list)
    """The series of actual flows, which is filled during the simulation."""

    def __post_init__(self):
        self._info['current_flow'] = None
        assert self.min_flow >= 0, f"The minimum flow cannot be negative, got {self.min_flow}"
        assert self.max_flow >= self.min_flow, \
            f"The maximum flow cannot be lower than the minimum, got {self.max_flow} < {self.min_flow}"
        assert len(self._destination.commodities_in) != 0, \
            f"Destination node '{self._destination.name}' does not accept any input commodity, but it should"
        assert self.commodity in self._source.commodities_out, \
            f"Source node '{self._source.name}' should return commodity '{self.commodity}', " \
            f"but it returns {self._source.commodities_out}"
        assert self.commodity in self._destination.commodities_in, \
            f"Destination node '{self._destination.name}' should accept commodity '{self.commodity}', " \
            f"but it accepts {self._destination.commodities_in}"

    @classproperty
    def _properties(self) -> List[str]:
        properties = super(Edge, self)._properties
        return properties + ['source', 'destination', 'commodity', 'min_flow', 'max_flow', 'bounds', 'current_flow']

    @property
    def source(self) -> str:
        """The source node."""
        return self._source.name

    @property
    def destination(self) -> str:
        """The destination node."""
        return self._destination.name

    @property
    def bounds(self) -> Tuple[float, float]:
        """The lower and upper flow bounds for the commodity."""
        return self.min_flow, self.max_flow

    @property
    def flows(self) -> pd.Series:
        """The series of actual flows, which is filled during the simulation."""
        return pd.Series(self._flows, dtype=float, index=self._horizon[:len(self._flows)])

    @property
    def current_flow(self) -> Optional[Flow]:
        """The current flow on the edge for this time step as provided by the user."""
        return self._info['current_flow']

    def to_pyomo(self, mutable: bool = False) -> pyo.Block:
        # build an edge block with a variable representing the flow
        edge = pyo.Block(concrete=True, name=self.name)
        # the variable is bounded by the min/max bounds and is initialized to the current flow if needed
        kwargs = dict() if mutable else dict(initialize=self.current_flow)
        edge.flow = pyo.Var(domain=pyo.NonNegativeReals, bounds=self.bounds, **kwargs)
        return edge

    def update(self, rng: np.random.Generator, flows: Dict[Any, Flow], states: Dict[Any, State]):
        self._info['current_flow'] = flows[self]

    def step(self, flows: Dict[Any, Flow], states: Dict[Any, State]):
        flow = flows[self]
        assert flow >= self.min_flow, f"Flow for edge {self.key} should be >= {self.min_flow}, got {flow}"
        assert flow <= self.max_flow, f"Flow for edge {self.key} should be <= {self.max_flow}, got {flow}"
        self._flows.append(flow)
        self._info['current_flow'] = None


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class SingleEdge(Edge):
    """An edge in a plant where two nodes can be connected by a unique edge (i.e., a graph)."""

    @property
    def key(self) -> SingleEdgeID:
        return self.source, self.destination

    @property
    def name(self) -> str:
        return f"{self.source} --> {self.destination}"


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class MultiEdge(Edge):
    """An edge in a plant where two nodes can be connected by a multiple edges (i.e., a multi-graph)."""

    @property
    def key(self) -> MultiEdgeID:
        return self.source, self.destination, self.commodity

    @property
    def name(self) -> str:
        return f"{self.source} --({self.commodity})--> {self.destination}"
