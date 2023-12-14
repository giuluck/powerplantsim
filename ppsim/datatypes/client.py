from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, Optional, List

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import VarianceNode


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
    def commodity_in(self) -> Optional[str]:
        return self.commodity

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
        properties = super(VarianceNode, self)._properties
        return properties + ['predicted_demands', 'demands']

    @property
    def predicted_demands(self) -> pd.Series:
        """The series of predicted demands."""
        return self.predictions

    @property
    def demands(self) -> pd.Series:
        """The series of actual demands, which is filled during the simulation."""
        return self.values

    def update(self, rng: np.random.Generator):
        index = self._step()
        value = self._predictions[index] + self._variance_fn(rng, self.values)
        flow = np.sum([e.flow_at(index=index) for e in self._in_edges])
        assert value <= flow, \
            f"Customer node '{self.name}' can accept at most {value} units at time step {index}, got {flow}"
        self._values[index] = value


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Purchaser(Client):
    """A node in the plant that buys a unique commodity."""

    @classproperty
    def purchaser(self) -> bool:
        return True

    @classproperty
    def _properties(self) -> List[str]:
        properties = super(VarianceNode, self)._properties
        return properties + ['predicted_prices', 'prices']

    @property
    def predicted_prices(self) -> pd.Series:
        """The series of predicted buying prices."""
        return self.predictions

    @property
    def prices(self) -> pd.Series:
        """The series of actual buying prices, which is filled during the simulation."""
        return self.values
