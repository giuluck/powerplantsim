from abc import ABC
from dataclasses import dataclass
from typing import Set, Optional

import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import VarianceNode, InternalVarianceNode


@dataclass(repr=False, eq=False, slots=True)
class Client(VarianceNode, ABC):
    """A node in the plant that buys/asks for a unique commodity and can be exposed to the user."""
    pass


@dataclass(repr=False, eq=False, slots=True)
class Customer(Client):
    """A node in the plant that asks for a unique commodity and can be exposed to the user."""

    @property
    def demands(self) -> pd.Series:
        """The series of predicted demands."""
        return self.predictions


@dataclass(repr=False, eq=False, slots=True)
class Purchaser(Client):
    """A node in the plant that buys a unique commodity and can be exposed to the user."""

    @property
    def prices(self) -> pd.Series:
        """The series of predicted buying prices."""
        return self.predictions


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class InternalClient(InternalVarianceNode, ABC):
    """A node in the plant that buys/asks for a unique commodity and is not exposed to the user."""

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
    @property
    def demands(self) -> pd.Series:
        """The series of predicted demands."""
        return self.predictions

    @property
    def exposed(self) -> Customer:
        return Customer(
            name=self.name,
            commodity_in=self.commodity_in,
            commodities_out=self.commodities_out,
            predictions=self.demands.copy(deep=True)
        )


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class InternalPurchaser(InternalClient):
    @property
    def prices(self) -> pd.Series:
        """The series of predicted buying prices."""
        return self.predictions

    @property
    def exposed(self) -> Purchaser:
        return Purchaser(
            name=self.name,
            commodity_in=self.commodity_in,
            commodities_out=self.commodities_out,
            predictions=self.prices.copy(deep=True)
        )
