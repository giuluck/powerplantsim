from dataclasses import dataclass, field
from typing import Callable, Set

import numpy as np
import pandas as pd

from ppsim.datatypes.node import Node, Nodes


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False)
class Client(Node):
    """A node in the plant that asks for a unique commodity."""

    commodity: str = field(kw_only=True)
    """The identifier of the commodity asked by the node."""

    demands: pd.Series = field(kw_only=True)
    """The series of (predicted) demands."""

    _variance: Callable[[np.random.Generator, pd.Series], float] = field(kw_only=True)
    """A function f(rng, series) -> variance describing the variance model of true demands."""

    kind: Nodes = field(init=False, kw_only=True, default=Nodes.CLIENT)
    commodity_in: bool = field(init=False, kw_only=True, default=True)
    commodity_out: bool = field(init=False, kw_only=True, default=False)

    @property
    def commodities_in(self) -> Set[str]:
        return {self.commodity}

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
        return self._variance(rng, series)
