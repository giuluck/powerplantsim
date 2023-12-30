import unittest
from abc import abstractmethod

import numpy as np
import pandas as pd

from test.utils import TestPlant

PLANT = TestPlant(horizon=3)

HORIZON = pd.Index(np.arange(3))

SERIES_1 = pd.Series([3.0, 2.0, 1.0])

SERIES_2 = pd.Series([-1.0, -2.0, -3.0])

VARIANCE_1 = lambda rng, series: rng.normal()

VARIANCE_2 = lambda rng, series: rng.random() - 0.5

SETPOINT = pd.DataFrame(
    data={
        ('input', 'in_com'): [100.0, 50.0, 75.0],
        ('output', 'out_com_1'): [1.0, 0.0, 0.5],
        ('output', 'out_com_2'): [60.0, 10.0, 30.0]
    },
    index=[1.0, 0.5, 0.75]
)


class TestDataType(unittest.TestCase):
    @abstractmethod
    def test_inputs(self):
        """Tests that sanity checks on the user input are correctly implemented."""
        pass

    @abstractmethod
    def test_hashing(self):
        """Tests that the object is correctly hashed for the equality checks."""
        pass

    @abstractmethod
    def test_properties(self):
        """Tests that the internal properties of a datatype were consistently stored."""
        pass

    @abstractmethod
    def test_immutability(self):
        """Tests that the internal mutable datatypes cannot be changed."""
        pass

    @abstractmethod
    def test_dict(self):
        """Tests that the correct dictionary of datatype properties is returned."""
        pass

    @abstractmethod
    def test_operation(self):
        """Tests that the datatype works correctly during the simulation."""
        pass

    @abstractmethod
    def test_pyomo(self):
        """Tests that the pyomo encoding of the node is correct."""
        pass
