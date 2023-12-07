import unittest

import pandas as pd

from ppsim.datatypes import Client, Supplier, Machine, Storage, Edge

SERIES_1 = pd.Series([3.0, 2.0, 1.0])
SERIES_2 = pd.Series([-1.0, -2.0, -3.0])
VARIANCE_1 = lambda rng, series: rng.normal()
VARIANCE_2 = lambda rng, series: rng.random() - 0.5
SETPOINT = pd.DataFrame(
    data={'out_com_1': [1.0, 0.0, 0.5], 'out_com_2': [60.0, 10.0, 30.0]},
    index=[100.0, 50.0, 75.0]
)


class TestDataTypes(unittest.TestCase):

    def test_supplier(self):
        s = Supplier(name='s', commodity='s_com', prices=SERIES_1, _variance=VARIANCE_1)
        # test properties
        self.assertEqual(s.key, 's', msg="Wrong supplier key name stored")
        self.assertEqual(s.kind, 'supplier', msg="Supplier node is not labelled as supplier")
        self.assertFalse(s.commodity_in, msg="Supplier node should not have input commodities")
        self.assertTrue(s.commodity_out, msg="Supplier node should have output commodities")
        self.assertSetEqual(s.commodities_in, set(), msg="Wrong supplier inputs stored")
        self.assertSetEqual(s.commodities_out, {'s_com'}, msg="Wrong supplier outputs stored")
        # test hashing
        s_equal = Supplier(name='s', commodity='s_com_2', prices=SERIES_2, _variance=VARIANCE_2)
        self.assertEqual(s, s_equal, msg="Nodes with the same name should be considered equal")
        s_diff = Supplier(name='sd', commodity='s_com', prices=SERIES_1, _variance=VARIANCE_1)
        self.assertNotEqual(s, s_diff, msg="Nodes with different names should be considered different")

    def test_client(self):
        c = Client(name='c', commodity='c_com', demands=SERIES_1, _variance=VARIANCE_1)
        # test properties
        self.assertEqual(c.key, 'c', msg="Wrong client key name stored")
        self.assertEqual(c.kind, 'client', msg="Client node is not labelled as client")
        self.assertTrue(c.commodity_in, msg="Client node should have input commodities")
        self.assertFalse(c.commodity_out, msg="Client node should not have output commodities")
        self.assertSetEqual(c.commodities_in, {'c_com'}, msg="Wrong client inputs stored")
        self.assertSetEqual(c.commodities_out, set(), msg="Wrong client outputs stored")
        # test hashing
        c_equal = Client(name='c', commodity='c_com_2', demands=SERIES_2, _variance=VARIANCE_2)
        self.assertEqual(c, c_equal, msg="Nodes with the same name should be considered equal")
        c_diff = Client(name='cd', commodity='c_com', demands=SERIES_1, _variance=VARIANCE_1)
        self.assertNotEqual(c, c_diff, msg="Nodes with different names should be considered different")

    def test_machine(self):
        m = Machine(name='m', commodity='in_com', setpoint=SETPOINT, discrete_setpoint=False, max_starting=None, cost=0)
        # test properties
        self.assertEqual(m.key, 'm', msg="Wrong machine key name stored")
        self.assertEqual(m.kind, 'machine', msg="Machine node is not labelled as machine")
        self.assertTrue(m.commodity_in, msg="Machine node should have input commodities")
        self.assertTrue(m.commodity_out, msg="Machine node should have output commodities")
        self.assertSetEqual(m.commodities_in, {'in_com'}, msg="Wrong machine inputs stored")
        self.assertSetEqual(m.commodities_out, {'out_com_1', 'out_com_2'}, msg="Wrong machine outputs stored")
        self.assertListEqual(list(m.setpoint.index), [50.0, 75.0, 100.0], msg="Wrong setpoint index stored")
        # test hashing
        m_equal = Machine(
            name='m',
            commodity='in_com_2',
            setpoint=SETPOINT.copy(deep=True),
            discrete_setpoint=True,
            max_starting=(1, 7),
            cost=100
        )
        self.assertEqual(m, m_equal, msg="Nodes with the same name should be considered equal")
        m_diff = Machine(
            name='md',
            commodity='in_com',
            setpoint=SETPOINT,
            discrete_setpoint=False,
            max_starting=None,
            cost=0
        )
        self.assertNotEqual(m, m_diff, msg="Nodes with different names should be considered different")
        # test sanity checks
        with self.assertRaises(AssertionError, msg="Negative cost should raise exception"):
            Machine(name='m', commodity='in_com', setpoint=SETPOINT, discrete_setpoint=True, max_starting=None, cost=-1)
        sp2 = SETPOINT.copy(deep=True)
        sp2.index = [50.0, 0.0, 100.0]
        with self.assertRaises(AssertionError, msg="Null flows in setpoint should raise exception"):
            Machine(name='m', commodity='in_com', setpoint=sp2, discrete_setpoint=False, max_starting=None, cost=0)
        sp2.index = [50.0, -1.0, 100.0]
        with self.assertRaises(AssertionError, msg="Negative flows in setpoint should raise exception"):
            Machine(name='m', commodity='in_com', setpoint=sp2, discrete_setpoint=False, max_starting=None, cost=0)
        # test operations on discrete setpoint
        m = Machine(name='m', commodity='in_com', setpoint=SETPOINT, discrete_setpoint=True, max_starting=None, cost=0)
        self.assertIsNone(m.operate(0.0), msg="Operate function should return None if the input is null")
        self.assertDictEqual(m.operate(50.0), {'out_com_1': 0.0, 'out_com_2': 10.0}, msg="Wrong operation output")
        self.assertDictEqual(m.operate(75.0), {'out_com_1': 0.5, 'out_com_2': 30.0}, msg="Wrong operation output")
        self.assertDictEqual(m.operate(100.0), {'out_com_1': 1.0, 'out_com_2': 60.0}, msg="Wrong operation output")
        with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-data flows"):
            m.operate(70.0)
        with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-data flows"):
            m.operate(90.0)
        with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-bound flows"):
            m.operate(20.0)
        with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-bound flows"):
            m.operate(110.0)
        # test operations on continuous setpoint
        m = Machine(name='m', commodity='in_com', setpoint=SETPOINT, discrete_setpoint=False, max_starting=None, cost=0)
        self.assertIsNone(m.operate(0.0), msg="Operate function should return None if the input is null")
        self.assertDictEqual(m.operate(50.0), {'out_com_1': 0.0, 'out_com_2': 10.0}, msg="Wrong operation output")
        self.assertDictEqual(m.operate(75.0), {'out_com_1': 0.5, 'out_com_2': 30.0}, msg="Wrong operation output")
        self.assertDictEqual(m.operate(100.0), {'out_com_1': 1.0, 'out_com_2': 60.0}, msg="Wrong operation output")
        self.assertDictEqual(m.operate(70.0), {'out_com_1': 0.4, 'out_com_2': 26.0}, msg="Wrong operation output")
        self.assertDictEqual(m.operate(90.0), {'out_com_1': 0.8, 'out_com_2': 48.0}, msg="Wrong operation output")
        with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-bound flows"):
            m.operate(20.0)
        with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-bound flows"):
            m.operate(110.0)

    def test_storage(self):
        s = Storage(name='s', commodity='s_com', capacity=100, dissipation=1.0)
        # test properties
        self.assertEqual(s.key, 's', msg="Wrong storage key name stored")
        self.assertEqual(s.kind, 'storage', msg="Storage node is not labelled as supplier")
        self.assertTrue(s.commodity_in, msg="Storage node should have input commodities")
        self.assertTrue(s.commodity_out, msg="Storage node should have output commodities")
        self.assertSetEqual(s.commodities_in, {'s_com'}, msg="Wrong storage inputs stored")
        self.assertSetEqual(s.commodities_out, {'s_com'}, msg="Wrong storage outputs stored")
        # test hashing
        s_equal = Storage(name='s', commodity='s_com_2', capacity=50, dissipation=1.0)
        self.assertEqual(s, s_equal, msg="Nodes with the same name should be considered equal")
        s_diff = Storage(name='sd', commodity='s_com', capacity=100, dissipation=1.0)
        self.assertNotEqual(s, s_diff, msg="Nodes with different names should be considered different")
        # test sanity checks
        Storage(name='s', commodity='s_com', capacity=100, dissipation=0.0)
        Storage(name='s', commodity='s_com', capacity=100, dissipation=0.5)
        Storage(name='s', commodity='s_com', capacity=100, dissipation=1.0)
        with self.assertRaises(AssertionError, msg="Out of bound dissipation should raise exception"):
            Storage(name='s', commodity='s_com', capacity=100, dissipation=-1.0)
        with self.assertRaises(AssertionError, msg="Out of bound dissipation should raise exception"):
            Storage(name='s', commodity='s_com', capacity=100, dissipation=2.0)
        with self.assertRaises(AssertionError, msg="Null capacity should raise exception"):
            Storage(name='s', commodity='s_com', capacity=0, dissipation=1.0)
        with self.assertRaises(AssertionError, msg="Negative capacity should raise exception"):
            Storage(name='s', commodity='s_com', capacity=-10, dissipation=1.0)

    def test_edge(self):
        m = Machine(name='m', commodity='in_com', setpoint=SETPOINT, discrete_setpoint=False, max_starting=None, cost=0)
        s1 = Storage(name='s1', commodity='out_com_1', capacity=100, dissipation=1.0)
        s2 = Storage(name='s2', commodity='out_com_2', capacity=100, dissipation=1.0)
        e = Edge(source=m, destination=s1, commodity='out_com_1', min_flow=0.0, max_flow=100.0, integer=False)
        # test properties
        self.assertIsInstance(e.key, Edge.EdgeID, msg="Wrong edge key type stored")
        self.assertTupleEqual(tuple(e.key), ('m', 's1', 'out_com_1'), msg="Wrong edge key stored")
        # test hashing
        e_equal = Edge(source=m, destination=s1, commodity='out_com_1', min_flow=50.0, max_flow=60.0, integer=True)
        self.assertEqual(e, e_equal, msg="Nodes with the same name should be considered equal")
        e_diff = Edge(source=m, destination=s2, commodity='out_com_2', min_flow=0.0, max_flow=100.0, integer=False)
        self.assertNotEqual(e, e_diff, msg="Nodes with different names should be considered different")
        # test sanity checks
        Edge(source=m, destination=s1, commodity='out_com_1', min_flow=0.0, max_flow=1.0, integer=False)
        Edge(source=m, destination=s1, commodity='out_com_1', min_flow=1.0, max_flow=2.0, integer=False)
        Edge(source=m, destination=s1, commodity='out_com_1', min_flow=1.0, max_flow=1.0, integer=False)
        with self.assertRaises(AssertionError, msg="Negative min flow should raise exception"):
            Edge(source=m, destination=s1, commodity='out_com_1', min_flow=-1.0, max_flow=100.0, integer=False)
        with self.assertRaises(AssertionError, msg="max flow < min flow should raise exception"):
            Edge(source=m, destination=s1, commodity='out_com_1', min_flow=101.0, max_flow=100.0, integer=False)
        with self.assertRaises(AssertionError, msg="Wrong source commodity should raise exception"):
            Edge(source=s1, destination=m, commodity='in_com', min_flow=0.0, max_flow=100.0, integer=False)
        with self.assertRaises(AssertionError, msg="Wrong destination commodity should raise exception"):
            Edge(source=s1, destination=m, commodity='out_com_1', min_flow=0.0, max_flow=100.0, integer=False)
        with self.assertRaises(AssertionError, msg="Wrong source and destination commodity should raise exception"):
            Edge(source=m, destination=s1, commodity='in_com', min_flow=0.0, max_flow=100.0, integer=False)
