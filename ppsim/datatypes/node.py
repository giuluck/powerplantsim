from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set, Optional, Callable, List, Tuple

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.datatype import DataType


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Node(DataType, ABC):
    """Data class for an abstract node in the plant."""

    name: str = field(kw_only=True)
    """The name of the node."""

    @classproperty
    @abstractmethod
    def kind(self) -> str:
        """The node type."""
        pass

    @classproperty
    def _properties(self) -> List[str]:
        return ['name', 'kind', 'commodity_in', 'commodities_out']

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
    def key(self) -> str:
        return self.name

    @property
    def _edges(self) -> Tuple[set, set]:
        """A tuple <in_edges, out_edges> of sets of Edge objects containing all the input/output edges of this node."""
        in_edges, out_edges = set(), set()
        for edge in self._plant.edges():
            if edge.source == self.name:
                out_edges.add(edge)
            elif edge.destination == self.name:
                in_edges.add(edge)
        return in_edges, out_edges

    def _instance(self, other) -> bool:
        return isinstance(other, Node)

    @abstractmethod
    def update(self, rng: np.random.Generator):
        """Updates the simulation by computing the internal values of the node.

        :param rng:
            The random number generator to be used for reproducible results.
        """
        pass


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class VarianceNode(Node, ABC):
    """A node in the plant that contains a series of predicted values and a variance model for a single commodity."""

    commodity: str = field(kw_only=True)
    """The identifier of the commodity handled by the node."""

    _predictions: np.ndarray = field(kw_only=True)
    """The series of predictions."""

    _variance_fn: Callable[[np.random.Generator, pd.Series], float] = field(kw_only=True)
    """A function f(rng, series) -> variance describing the variance model of true values."""

    _values: List[float] = field(init=False, default_factory=list)
    """The series of actual values, which is filled during the simulation."""

    def __post_init__(self):
        assert len(self._predictions) == len(self._horizon), \
            f"Predictions should match length of horizon, got {len(self._predictions)} instead of {len(self._horizon)}"

    @property
    def predictions(self) -> pd.Series:
        """The series of predictions."""
        return pd.Series(self._predictions.copy(), dtype=float, index=self._horizon)

    @property
    def values(self) -> pd.Series:
        """The series of actual values, which is filled during the simulation."""
        return pd.Series(self._values.copy(), dtype=float, index=self._horizon[:self._info.step + 1])

    def update(self, rng: np.random.Generator):
        # compute the new values as the sum of the prediction and the variance obtained from the variance model
        step = self._step()
        values = pd.Series(self._values.copy(), dtype=float, index=self._horizon[:self._info.step])
        next_value = self._predictions[step] + self._variance_fn(rng, values)
        self._values.append(next_value)
