from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, Optional

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import VarianceNode, InternalVarianceNode


@dataclass(repr=False, eq=False, slots=True)
class Client(VarianceNode, ABC):
    """A node in the plant that buys/asks for a unique commodity and can be exposed to the user."""

    @classproperty
    @abstractmethod
    def purchaser(self) -> bool:
        """Whether the client node buys the commodity at a given price (series = prices) or requires that an exact
        amount of commodities is sent to it according to its demands (series = demands)."""
        pass


@dataclass(repr=False, eq=False, slots=True)
class Customer(Client):
    """A node in the plant that asks for a unique commodity and can be exposed to the user."""

    @classproperty
    def purchaser(self) -> bool:
        return False

    @property
    def demands(self) -> pd.Series:
        """The series of predicted demands (alias for predictions)."""
        return self.predictions


@dataclass(repr=False, eq=False, slots=True)
class Purchaser(Client):
    """A node in the plant that buys a unique commodity and can be exposed to the user."""

    @classproperty
    def purchaser(self) -> bool:
        return True

    @property
    def prices(self) -> pd.Series:
        """The series of predicted buying prices (alias for predictions)."""
        return self.predictions


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class InternalClient(InternalVarianceNode, ABC):
    """A node in the plant that buys/asks for a unique commodity and is not exposed to the user."""

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
class InternalCustomer(InternalClient):
    """A node in the plant that asks for a unique commodity and is not exposed to the user."""

    @classproperty
    def purchaser(self) -> bool:
        return False

    @property
    def exposed(self) -> Customer:
        return Customer(
            name=self.name,
            commodity_in=self.commodity_in,
            commodities_out=self.commodities_out,
            predictions=self._predictions.copy()
        )

    def update(self, rng: np.random.Generator):
        index = self._step()
        value = self._predictions[index] + self._variance_fn(rng, self.values)
        flow = np.sum([e.flow_at(index=index) for e in self._in_edges])
        assert value <= flow, \
            f"Customer node '{self.name}' can accept at most {value} units at time step {index}, got {flow}"
        self._values[index] = value


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class InternalPurchaser(InternalClient):
    """A node in the plant that buys a unique commodity and is not exposed to the user."""

    @classproperty
    def purchaser(self) -> bool:
        return True

    @property
    def exposed(self) -> Purchaser:
        return Purchaser(
            name=self.name,
            commodity_in=self.commodity_in,
            commodities_out=self.commodities_out,
            predictions=self._predictions.copy()
        )
