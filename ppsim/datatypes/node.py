from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set, Callable, List, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
# noinspection PyPackageRequirements
from descriptors import classproperty

from ppsim.datatypes.datatype import DataType
from ppsim.utils.typing import NodeID, Flows, States


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Node(DataType, ABC):
    """Data class for an abstract node in the plant."""

    name: str = field(kw_only=True)
    """The name of the datatype."""

    @classproperty
    @abstractmethod
    def kind(self) -> str:
        """The node type."""
        pass

    @classproperty
    def _properties(self) -> List[str]:
        return ['name', 'kind']

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
    def key(self) -> NodeID:
        return self.name

    @property
    def _edges(self) -> Tuple[set, set]:
        """A tuple <in_edges, out_edges> of sets of Edge objects containing all the input/output edges of this node."""
        in_edges, out_edges = set(), set()
        for edge in self._plant.edges().values():
            if edge.source == self.name:
                out_edges.add(edge)
            elif edge.destination == self.name:
                in_edges.add(edge)
        return in_edges, out_edges

    def _instance(self, other) -> bool:
        return isinstance(other, Node)

    # noinspection PyUnresolvedReferences
    def to_pyomo(self, mutable: bool = False) -> pyo.Block:
        # build a node block with two variable arrays representing the input/output flows indexed by commodity
        node = pyo.Block(concrete=True, name=self.name)
        # these flows are not bounded but they must be constrained to be equal to the sum of edge variables
        node.in_flows = pyo.Var(self.commodities_in, domain=pyo.NonNegativeReals)
        node.out_flows = pyo.Var(self.commodities_out, domain=pyo.NonNegativeReals)
        return node


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class VarianceNode(Node, ABC):
    """A node in the plant that contains a series of predicted values and a variance model for a single commodity."""

    commodity: str = field(kw_only=True)
    """The identifier of the commodity handled by the node."""

    _predictions: np.ndarray = field(kw_only=True)
    """The series of predictions."""

    _variance_fn: Callable[[np.random.Generator, pd.Series], float] = field(kw_only=True)
    """A function f(rng, series) -> variance describing the variance model of true values."""

    _values: List[float] = field(init=False, default_factory=list)
    """The series of actual values, which is filled during the simulation."""

    def __post_init__(self):
        self._info['current_value'] = None
        assert len(self._predictions) == len(self._horizon), \
            f"Predictions should match length of horizon, got {len(self._predictions)} instead of {len(self._horizon)}"

    @classproperty
    def _properties(self) -> List[str]:
        properties = super(VarianceNode, self)._properties
        return properties + ['commodity']

    @property
    def values(self) -> pd.Series:
        """The series of actual values, which is filled during the simulation."""
        return pd.Series(self._values.copy(), dtype=float, index=self._horizon[:len(self._values)])

    @property
    def current_value(self) -> float:
        """The current value of the node for this time step as computed using the variance model."""
        return self._info['current_value']

    def update(self, rng: np.random.Generator, flows: Flows, states: States):
        # compute the new value as the sum of the prediction and the variance obtained from the variance model
        self._info['current_value'] = self._predictions[self._step] + self._variance_fn(rng, self.values)
