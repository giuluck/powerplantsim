from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ppsim import utils


@dataclass(repr=False, eq=False, slots=True)
class DataType(ABC):
    """Abstract class that defines a datatype which can be exposed to the user."""

    def __eq__(self, other: Any) -> bool:
        # if the two classes are not exactly the same, return False
        # (use the '==' operator instead of the 'instanceof' operator to avoid subclasses matching)
        if self.__class__ != other.__class__:
            return False
        # continue with checking equality among all the parameters of the dataclass
        # (implement specific checks for pandas datatypes)
        for param in self.__slots__:
            p1 = getattr(self, param)
            p2 = getattr(other, param)
            # COMPUTE EQUALITY
            if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
                # for numpy arrays, check equal shape first, then equality among values
                equal = p1.shape == p2.shape and np.all(p1 == p2)
            elif isinstance(p1, pd.Index) and isinstance(p2, pd.Index):
                # for pandas indices, check equal shape first, then equality among values
                equal = p1.shape == p2.shape and np.all(p1.values == p2.values)
            elif isinstance(p1, pd.Series) and isinstance(p2, pd.Series):
                # for pandas series, check equal shape first, then equality among index and values
                equal = p1.shape == p2.shape and np.all(p1.index == p2.index) and np.all(p1.values == p2.values)
            elif isinstance(p1, pd.DataFrame) and isinstance(p2, pd.DataFrame):
                # for pandas dataframes, check equal shape first, then equality among index, column, and values
                equal = p1.shape == p2.shape and \
                        np.all(p1.index == p2.index) and \
                        np.all(p1.columns == p2.columns) and \
                        np.all(p1.values == p2.values)
            else:
                # for different object types, check that the class is the same then check for equality
                equal = p1.__class__ == p2.__class__ and p1 == p2
            # CHECK EQUALITY
            # if the two objects are not equal, stop the check
            if not equal:
                return False
        # if the loop was not stopped, the two objects are equal
        return True


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class InternalDataType(ABC):
    """Abstract class that defines a datatype which is not exposed to the user and has a unique key for comparison."""

    @property
    @abstractmethod
    def key(self) -> Any:
        """An identifier of the object."""
        pass

    @property
    @abstractmethod
    def exposed(self) -> DataType:
        """The public datatype that can be exposed to the user."""
        pass

    def _instance(self, other) -> bool:
        """Checks whether a different object is matching the self instance for comparison."""
        return isinstance(other, self.__class__)

    def __eq__(self, other: Any) -> bool:
        return self._instance(other) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({utils.stringify(value=self.key)})"
