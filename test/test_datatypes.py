import unittest

import pandas as pd

from ppsim.datatypes import Client, Nodes, Supplier, Machine

SERIES_1 = pd.Series([3.0, 2.0, 1.0])
SERIES_2 = pd.Series([-1.0, -2.0, -3.0])
VARIANCE_1 = lambda rng, series: rng.normal()
VARIANCE_2 = lambda rng, series: rng.random() - 0.5


class TestDataTypes(unittest.TestCase):

    def test_supplier(self):
        s = Supplier(name='sup', commodity='sup_com', prices=SERIES_1, _variance=VARIANCE_1)
        # test properties
        self.assertEqual(s.key, 'sup', msg="Wrong supplier key name stored")
        self.assertEqual(s.kind, Nodes.SUPPLIER, msg="Supplier node is not labelled as supplier")
        self.assertFalse(s.commodity_in, msg="Supplier node should not have input commodities")
        self.assertTrue(s.commodity_out, msg="Supplier node should have output commodities")
        self.assertSetEqual(s.commodities_in, set(), msg="Wrong supplier inputs stored")
        self.assertSetEqual(s.commodities_out, {'sup_com'}, msg="Wrong supplier outputs stored")
        # test hashing
        s_equal = Supplier(name='sup', commodity='sup_com_2', prices=SERIES_2, _variance=VARIANCE_2)
        self.assertEqual(s, s_equal, msg="Nodes with the same name should be considered equal")
        s_diff = Supplier(name='sup_diff', commodity='sup_com', prices=SERIES_1, _variance=VARIANCE_1)
        self.assertNotEqual(s, s_diff, msg="Nodes with different names should be considered different")

    def test_client(self):
        c = Client(name='cli', commodity='cli_com', demands=SERIES_1, _variance=VARIANCE_1)
        # test properties
        self.assertEqual(c.key, 'cli', msg="Wrong client key name stored")
        self.assertEqual(c.kind, Nodes.CLIENT, msg="Client node is not labelled as client")
        self.assertTrue(c.commodity_in, msg="Client node should have input commodities")
        self.assertFalse(c.commodity_out, msg="Client node should not have output commodities")
        self.assertSetEqual(c.commodities_in, {'cli_com'}, msg="Wrong client inputs stored")
        self.assertSetEqual(c.commodities_out, set(), msg="Wrong client outputs stored")
        # test hashing
        c_equal = Client(name='cli', commodity='cli_com_2', demands=SERIES_2, _variance=VARIANCE_2)
        self.assertEqual(c, c_equal, msg="Nodes with the same name should be considered equal")
        c_diff = Client(name='cli_diff', commodity='cli_com', demands=SERIES_1, _variance=VARIANCE_1)
        self.assertNotEqual(c, c_diff, msg="Nodes with different names should be considered different")

    def test_machine(self):
        sp = pd.DataFrame(
            data={'out_com_1': [1.0, 0.0, 0.5], 'out_com_2': [60.0, 10.0, 30.0]},
            index=[100.0, 50.0, 75.0]
        )
        m = Machine(name='mac', commodity='in_com', setpoint=sp)
        # test properties
        self.assertEqual(m.key, 'mac', msg="Wrong machine key name stored")
        self.assertEqual(m.kind, Nodes.MACHINE, msg="Machine node is not labelled as machine")
        self.assertTrue(m.commodity_in, msg="Machine node should have input commodities")
        self.assertTrue(m.commodity_out, msg="Machine node should have output commodities")
        self.assertSetEqual(m.commodities_in, {'in_com'}, msg="Wrong machine inputs stored")
        self.assertSetEqual(m.commodities_out, {'out_com_1', 'out_com_2'}, msg="Wrong machine outputs stored")
        self.assertListEqual(list(m.setpoint.index), [50.0, 75.0, 100.0], msg="Wrong setpoint index stored")
        # test hashing
        m_equal = Machine(name='mac', commodity='in_com_2', setpoint=sp.copy(deep=True))
        self.assertEqual(m, m_equal, msg="Nodes with the same name should be considered equal")
        m_diff = Machine(name='mac_diff', commodity='in_com', setpoint=sp)
        self.assertNotEqual(m, m_diff, msg="Nodes with different names should be considered different")
        # test sanity checks
        sp2 = sp.copy(deep=True)
        sp2.index = [50.0, 0.0, 100.0]
        with self.assertRaises(AssertionError, msg="Null flows in setpoint should raise exception"):
            Machine(name='mac', commodity='in_com', setpoint=sp2)
        sp2.index = [50.0, -1.0, 100.0]
        with self.assertRaises(AssertionError, msg="Negative flows in setpoint should raise exception"):
            Machine(name='mac', commodity='in_com', setpoint=sp2)
        # test operations on discrete setpoint
        m = Machine(name='mac', commodity='in_com', setpoint=sp, discrete_setpoint=True)
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
        m = Machine(name='mac', commodity='in_com', setpoint=sp, discrete_setpoint=False)
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
        self.fail(msg='Not implemented yet')
