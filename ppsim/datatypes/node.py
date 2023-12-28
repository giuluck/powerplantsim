from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set, List, Tuple

import pyomo.environ as pyo
# noinspection PyPackageRequirements
from descriptors import classproperty

from ppsim.datatypes.datatype import DataType
from ppsim.utils.typing import NodeID


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
        properties = super(Node, self)._properties
        return properties + ['kind']

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
