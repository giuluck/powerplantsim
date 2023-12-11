from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set, Optional, Callable

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.datatype import InternalDataType, DataType


@dataclass(repr=False, eq=False, slots=True)
class Node(DataType):
    """Public data class for an abstract node in the plant which can be exposed to the user."""

    name: str = field(kw_only=True, repr=True)
    """The name of the node."""

    commodity_in: Optional[str] = field(kw_only=True)
    """The (optional) input commodities that is accepted (can be at most one)."""

    commodities_out: Set[str] = field(kw_only=True)
    """The set of output commodities that is returned (can be more than one)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


@dataclass(repr=False, eq=False, slots=True)
class VarianceNode(Node, ABC):
    """A node in the plant that contains a series of predicted values and a variance model for a single commodity and
    can be exposed to the user."""

    predictions: pd.Series = field(kw_only=True)
    """The series of predictions."""


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class InternalNode(InternalDataType, ABC):
    """Data class for an abstract node in the plant which is not exposed to the user."""

    name: str = field(kw_only=True)
    """The name of the node."""

    parents: list = field(kw_only=True, init=False, default_factory=lambda: [])
    """A list containing all the parent nodes."""

    children: list = field(kw_only=True, init=False, default_factory=lambda: [])
    """A list containing all the children nodes."""

    @classproperty
    @abstractmethod
    def kind(self) -> str:
        """The node type."""
        pass

    @property
    @abstractmethod
    def commodity_in(self) -> Optional[str]:
        """The (optional) input commodities that is accepted (can be at most one)."""
        pass

    @property
    @abstractmethod
    def commodities_out(self) -> Set[str]:
        """The set of output commodities that is returned (can be more than one)."""
        pass

    @property
    @abstractmethod
    def exposed(self) -> Node:
        pass

    @property
    def key(self) -> str:
        return self.name

    def _instance(self, other) -> bool:
        return isinstance(other, InternalNode)


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class InternalVarianceNode(InternalNode, ABC):
    """A node in the plant that contains a series of predicted values and a variance model for a single commodity and
    is not exposed to the user."""

    commodity: str = field(kw_only=True)
    """The identifier of the commodity handled by the node."""

    predictions: pd.Series = field(kw_only=True)
    """The series of predictions."""

    variance_fn: Callable[[np.random.Generator, pd.Series], float] = field(kw_only=True)
    """A function f(rng, series) -> variance describing the variance model of true values."""

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
        return self.variance_fn(rng, series)
