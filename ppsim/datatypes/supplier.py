from dataclasses import dataclass, field
from typing import Callable, Set, Optional

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import InternalNode, Node


@dataclass()
class Supplier(Node):
    """A node in the plant that can supply a unique commodity and can be exposed to the user."""

    prices: pd.Series = field(kw_only=True)
    """The series of (predicted) prices."""


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False)
class InternalSupplier(InternalNode):
    """A node in the plant that can supply a unique commodity and is not exposed to the user."""

    commodity: str = field(kw_only=True)
    """The identifier of the commodity supplied by the node."""

    prices: pd.Series = field(kw_only=True)
    """The series of (predicted) prices."""

    variance_fn: Callable[[np.random.Generator, pd.Series], float] = field(kw_only=True)
    """A function f(rng, series) -> variance describing the variance model of true prices."""

    @classproperty
    def kind(self) -> str:
        return 'supplier'

    @property
    def commodity_in(self) -> Optional[str]:
        return None

    @property
    def commodities_out(self) -> Set[str]:
        return {self.commodity}

    def variance(self, rng: np.random.Generator, series: pd.Series) -> float:
        """Describes the variance model of the array of true prices with respect to the predictions.

        :param rng:
            The random number generator to be used for reproducible results.

        :param series:
            The time series of previous true values indexed by the datetime information passed to the plant.
            The datetime information can be either an integer representing the index in the series, or a more specific
            information which was passed to the plant.
            The last value of the series is always a nan value, and it will be computed at the successive iteration
            based on the respective predicted price and the output of this function.

        :return:
            A real number <eps> which represents the delta between the predicted price and the true price.
            For an input series with length L, the true price will be eventually computed as:
                true = self.prices[L] + <eps>
        """
        return VARIANCE_fn(rng, series)

    @property
    def exposed(self) -> Supplier:
        return Supplier(
            name=self.name,
            commodity_in=self.commodity_in,
            commodities_out=self.commodities_out,
            prices=self.prices.copy(deep=True)
        )
