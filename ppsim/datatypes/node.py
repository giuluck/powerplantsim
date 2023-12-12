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

    _in_edges: list = field(kw_only=True, init=False, default_factory=list)
    """A list of InternalEdge objects containing all the input edges (<source>, self)."""

    _out_edges: list = field(kw_only=True, init=False, default_factory=list)
    """A list  of InternalEdge objects containing all the output edges (self, <destination>)."""

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

    @abstractmethod
    def update(self, rng: np.random.Generator):
        """Updates the simulation by computing the internal values of the node.

        :param rng:
            The random number generator to be used for reproducible results.
        """
        pass

    def append(self, edge):
        """Appends an edge into the internal data structure.

        :param edge:
            The edge to append.
        """
        if edge.source.name == self.name:
            self._out_edges.append(edge)
        elif edge.destination.name == self.name:
            self._in_edges.append(edge)
        else:
            raise AssertionError(f"Trying to append {edge} to {self}, but the node is neither source nor destination")


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class InternalVarianceNode(InternalNode, ABC):
    """A node in the plant that contains a series of predicted values and a variance model for a single commodity and
    is not exposed to the user."""

    commodity: str = field(kw_only=True)
    """The identifier of the commodity handled by the node."""

    _predictions: pd.Series = field(kw_only=True)
    """The series of predictions."""

    _variance_fn: Callable[[np.random.Generator, pd.Series], float] = field(kw_only=True)
    """A function f(rng, series) -> variance describing the variance model of true values."""

    _values: pd.Series = field(init=False, default_factory=lambda: pd.Series(dtype=float))
    """The series of actual values, which is filled during the simulation."""

    @property
    def _horizon(self) -> pd.Index:
        return self._predictions.index

    @property
    def predictions(self) -> pd.Series:
        """The series of predictions."""
        return self._predictions.copy()

    @property
    def values(self) -> pd.Series:
        """The series of actual values, which is filled during the simulation."""
        return pd.Series(self._values, index=self._horizon, dtype=float)

    def update(self, rng: np.random.Generator):
        # compute the new values as the sum of the prediction and the variance obtained from the variance model
        index = self._step()
        value = self._predictions[index] + self._variance_fn(rng, self.values)
        self._values[index] = value
