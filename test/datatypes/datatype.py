import unittest
from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from ppsim.datatypes import MultiEdge, Storage, Node
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


def dummy_node(commodity: str, name: str = 'dummy'):
    return Storage(
        name=name,
        commodity=commodity,
        dissipation=0.0,
        capacity=100,
        charge_rate=100,
        discharge_rate=100,
        _plant=None
    )


def dummy_edge(commodity: str, source: Union[str, Node] = 'dummy', destination: Union[str, Node] = 'dummy'):
    return MultiEdge(
        _source=dummy_node(commodity=commodity, name=source) if isinstance(source, str) else source,
        _destination=dummy_node(commodity=commodity, name=destination) if isinstance(destination, str) else destination,
        commodity=commodity,
        min_flow=0,
        max_flow=100,
        _plant=None
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
