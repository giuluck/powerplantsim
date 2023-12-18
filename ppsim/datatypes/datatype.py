import copy
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim import utils
from ppsim.utils.typing import Flows, States


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class DataType(ABC):
    """Abstract class that defines a datatype and has a unique key for comparison."""

    _plant: Any = field(kw_only=True)
    """The power plant object to which this datatype is attached."""

    _info: Dict[str, Any] = field(init=False, default_factory=dict)
    """Internal object for additional mutable information."""

    @classproperty
    @abstractmethod
    def _properties(self) -> List[str]:
        """The list of public properties of the datatype."""
        pass

    @property
    @abstractmethod
    def key(self) -> Any:
        """An identifier of the object."""
        pass

    @property
    def _step(self) -> Optional[int]:
        """The current step of the simulation, or None if the simulation is not started yet."""
        return self._plant.step

    @property
    def _index(self) -> Optional:
        """The current index of the simulation as in the time horizon, or None if the simulation is not started."""
        return self._plant.index

    @property
    def _horizon(self) -> pd.Index:
        """The time horizon of the simulation in which the datatype is involved."""
        return self._plant.horizon

    @property
    def dict(self) -> Dict[str, Any]:
        """A dictionary containing all the information of the datatype object indexed via property name."""
        return {param: getattr(self, param) for param in self._properties}

    def to_json(self) -> Dict[str, Any]:
        """Function to make the object serializable.

        :return:
            A dictionary containing all the information of the datatype which can be dumped in a json file.
        """
        json = {}
        for param in self._properties:
            value = getattr(self, param)
            if isinstance(value, set):
                value = list(value)
            elif isinstance(value, pd.Series):
                value = value.to_dict()
            elif isinstance(value, pd.DataFrame):
                value = value.to_dict(orient='tight')
            json[param] = value
        return json

    def copy(self):
        """Copies the datatype.

        :return:
            A copy of the datatype.
        """
        return copy.deepcopy(self)

    def _instance(self, other) -> bool:
        """Checks whether a different object is matching the self instance for comparison."""
        return isinstance(other, self.__class__)

    @abstractmethod
    def update(self, rng: np.random.Generator, flows: Flows, states: States):
        """Updates the current internal values of the datatype before the recourse action is called.

        :param rng:
            The random number generator to be used for reproducible results.

        :param flows:
            The user-defined edge flows for the current step.

        :param states:
            The user-defined machine states for the current step.
        """
        pass

    @abstractmethod
    def step(self, flows: Flows, states: States):
        """Performs a step forward in the simulation by computing internal values after the recourse action is called.

        :param flows:
            The edge flows computed by the recourse action for the current step.

        :param states:
            The machine states computed by the recourse action for the current step.
        """
        pass

    def __eq__(self, other: Any) -> bool:
        return self._instance(other) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({utils.stringify(value=self.key)})"
