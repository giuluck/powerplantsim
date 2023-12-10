from dataclasses import dataclass, field
from typing import Callable, Set, Optional

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import InternalNode, Node


@dataclass()
class Client(Node):
    """A node in the plant that asks for a unique commodity and can be exposed to the user."""

    demands: pd.Series = field(kw_only=True)
    """The series of (predicted) demands."""


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class InternalClient(InternalNode):
    """A node in the plant that asks for a unique commodity and is not exposed to the user.."""

    commodity: str = field(kw_only=True)
    """The identifier of the commodity asked by the node."""

    demands: pd.Series = field(kw_only=True)
    """The series of (predicted) demands."""

    variance_fn: Callable[[np.random.Generator, pd.Series], float] = field(kw_only=True)
    """A function f(rng, series) -> variance describing the variance model of true demands."""

    @classproperty
    def kind(self) -> str:
        return 'client'

    @property
    def commodity_in(self) -> Optional[str]:
        return self.commodity

    @property
    def commodities_out(self) -> Set[str]:
        return set()

    def variance(self, rng: np.random.Generator, series: pd.Series) -> float:
        """Describes the variance model of the array of true demands with respect to the predictions.

        :param rng:
            The random number generator to be used for reproducible results.

        :param series:
            The time series of previous true values indexed by the datetime information passed to the plant.
            The datetime information can be either an integer representing the index in the series, or a more specific
            information which was passed to the plant.
            The last value of the series is always a nan value, and it will be computed at the successive iteration
            based on the respective predicted demand and the output of this function.

        :return:
            A real number <eps> which represents the delta between the predicted demand and the true demand.
            For an input series with length L, the true demand will be eventually computed as:
                true = self.demands[L] + <eps>
        """
        return VARIANCE_fn(rng, series)

    @property
    def exposed(self) -> Client:
        return Client(
            name=self.name,
            commodity_in=self.commodity_in,
            commodities_out=self.commodities_out,
            demands=self.demands.copy(deep=True)
        )
