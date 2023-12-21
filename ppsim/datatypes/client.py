from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, List, Optional

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import VarianceNode
from ppsim.utils.typing import Flows, States


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Client(VarianceNode, ABC):
    """A node in the plant that buys/asks for a unique commodity."""

    @classproperty
    @abstractmethod
    def purchaser(self) -> bool:
        """Whether the client node buys the commodity at a given price (series = prices) or requires that an exact
        amount of commodities is sent to it according to its demands (series = demands)."""
        pass

    @classproperty
    def kind(self) -> str:
        return 'client'

    @property
    def commodities_in(self) -> Set[str]:
        return {self.commodity}

    @property
    def commodities_out(self) -> Set[str]:
        return set()


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Customer(Client):
    """A node in the plant that asks for a unique commodity."""

    @classproperty
    def purchaser(self) -> bool:
        return False

    @classproperty
    def _properties(self) -> List[str]:
        properties = super(Customer, self)._properties
        return properties + ['purchaser', 'demands', 'current_demand']

    @property
    def demands(self) -> pd.Series:
        """The series of actual demands, which is filled during the simulation."""
        return self.values

    @property
    def current_demand(self) -> Optional[float]:
        """The current demand of the node for this time step as computed using the variance model."""
        return self.current_value

    def step(self, flows: Flows, states: States):
        # check that the flow does not exceed the demand
        demand = self._info['current_value']
        self._info['current_value'] = None
        flow = np.sum([flow for (_, destination, _), flow in flows.items() if destination == self.name])
        assert flow <= demand, f"Customer node '{self.name}' can accept at most {demand} units, got {flow}"
        self._values.append(demand)


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Purchaser(Client):
    """A node in the plant that buys a unique commodity."""

    @classproperty
    def purchaser(self) -> bool:
        return True

    @classproperty
    def _properties(self) -> List[str]:
        properties = super(Purchaser, self)._properties
        return properties + ['purchaser', 'prices', 'current_price']

    @property
    def prices(self) -> pd.Series:
        """The series of actual buying prices, which is filled during the simulation."""
        return self.values

    @property
    def current_price(self) -> Optional[float]:
        """The current buying price of the node for this time step as computed using the variance model."""
        return self.current_value

    def step(self, flows: Flows, states: States):
        self._values.append(self._info['current_value'])
        self._info['current_value'] = None
