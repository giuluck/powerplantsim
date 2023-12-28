from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Set, List, Optional

import numpy as np
import pandas as pd
import pyomo.environ as pyo
# noinspection PyPackageRequirements
from descriptors import classproperty

from ppsim.datatypes.node import Node
from ppsim.utils.typing import Flows, States


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class ExtremityNode(Node, ABC):
    """A node at the plant extremities that contains a series of values and a variance model for a single commodity."""

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
        properties = super(ExtremityNode, self)._properties
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


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class Client(ExtremityNode, ABC):
    """A node in the plant that buys/asks for a unique commodity."""

    @property
    def commodities_in(self) -> Set[str]:
        return {self.commodity}

    @property
    def commodities_out(self) -> Set[str]:
        return set()


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class Priced(ExtremityNode, ABC):
    """A node in the plant that buys/sells a unique commodity."""

    @classproperty
    def _properties(self) -> List[str]:
        properties = super(Priced, self)._properties
        return properties + ['current_price']

    @property
    @abstractmethod
    def _direction(self) -> int:
        """Whether the node sells the commodity (+1) or buys it (-1)."""
        pass

    @property
    def prices(self) -> pd.Series:
        """The series of actual buying prices, which is filled during the simulation."""
        return self.values

    @property
    def current_price(self) -> Optional[float]:
        """The current buying price of the node for this time step as computed using the variance model."""
        return self.current_value

    # noinspection PyUnresolvedReferences, PyTypeChecker
    def to_pyomo(self, mutable: bool = False) -> pyo.Block:
        # start from the default node block
        node = super(Priced, self).to_pyomo(mutable=mutable)
        # add a parameter representing the current price (and initialize it if needed)
        kwargs = dict(mutable=True) if mutable else dict(initialize=self.current_price)
        node.current_price = pyo.Param(domain=pyo.NonNegativeReals, **kwargs)
        # compute the cost from the input flow for the (unique) commodity and the price
        node.cost = self._direction * node.current_price * node.in_flows[self.commodity]
        return node

    def step(self, flows: Flows, states: States):
        self._values.append(self._info['current_value'])
        self._info['current_value'] = None


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class Customer(Client):
    """A node in the plant that asks for a unique commodity."""

    @classproperty
    def kind(self) -> str:
        return 'customer'

    @classproperty
    def _properties(self) -> List[str]:
        properties = super(Customer, self)._properties
        return properties + ['current_demand']

    @property
    def demands(self) -> pd.Series:
        """The series of actual demands, which is filled during the simulation."""
        return self.values

    @property
    def current_demand(self) -> Optional[float]:
        """The current demand of the node for this time step as computed using the variance model."""
        return self.current_value

    # noinspection PyUnresolvedReferences
    def to_pyomo(self, mutable: bool = False) -> pyo.Block:
        # start from the default node block
        node = super(Customer, self).to_pyomo(mutable=mutable)
        # add a parameter representing the current demand (and initialize it if needed)
        kwargs = dict(mutable=True) if mutable else dict(initialize=self.current_demand)
        node.current_demand = pyo.Param(domain=pyo.NonNegativeReals, **kwargs)
        # constraint the demand so that it matches the input flow of the (unique) commodity
        node.satisfy_demand = pyo.Constraint(rule=node.current_demand == node.in_flows[self.commodity])
        return node

    def step(self, flows: Flows, states: States):
        # check that the flow does not exceed the demand
        demand = self._info['current_value']
        self._info['current_value'] = None
        flow = np.sum([flow for (_, destination, _), flow in flows.items() if destination == self.name])
        assert flow <= demand, f"Customer node '{self.name}' can accept at most {demand} units, got {flow}"
        # assert np.isclose(flow, demand), f"Customer node '{self.name}' needs {demand} units, got {flow}"
        self._values.append(demand)


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class Purchaser(Client, Priced):
    """A node in the plant that buys a unique commodity."""

    def __post_init__(self):
        super(Purchaser, self).__post_init__()

    @classproperty
    def kind(self) -> str:
        return 'purchaser'

    @property
    def _direction(self) -> int:
        return -1


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class Supplier(Priced):
    """A node in the plant that can supply a unique commodity."""

    @classproperty
    def kind(self) -> str:
        return 'supplier'

    @property
    def _direction(self) -> int:
        return 1

    @property
    def commodities_in(self) -> Set[str]:
        return set()

    @property
    def commodities_out(self) -> Set[str]:
        return {self.commodity}
