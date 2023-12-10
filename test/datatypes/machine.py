from ppsim.datatypes import InternalMachine, Machine
from test.datatypes.datatype import TestDataType


class TestMachine(TestDataType):
    MACHINE = InternalMachine(
        name='m',
        commodity='in_com',
        setpoint=TestDataType.SETPOINT,
        discrete_setpoint=False,
        max_starting=None,
        cost=0
    )

    def test_checks(self):
        # test incorrect cost
        with self.assertRaises(AssertionError, msg="Negative cost should raise exception"):
            InternalMachine(
                name='m',
                commodity='in_com',
                setpoint=self.SETPOINT,
                discrete_setpoint=True,
                max_starting=None,
                cost=-1
            )
        # test incorrect setpoint
        sp2 = self.SETPOINT.copy(deep=True)
        sp2.index = [50.0, 0.0, 100.0]
        with self.assertRaises(AssertionError, msg="Null flows in setpoint should raise exception"):
            InternalMachine(
                name='m',
                commodity='in_com',
                setpoint=sp2,
                discrete_setpoint=False,
                max_starting=None,
                cost=0
            )
        sp2.index = [50.0, -1.0, 100.0]
        with self.assertRaises(AssertionError, msg="Negative flows in setpoint should raise exception"):
            InternalMachine(
                name='m',
                commodity='in_com',
                setpoint=sp2,
                discrete_setpoint=False,
                max_starting=None,
                cost=0
            )

    def test_hashing(self):
        # test equal hash
        m_equal = InternalMachine(
            name='m',
            commodity='in_com_2',
            setpoint=self.SETPOINT.copy(deep=True),
            discrete_setpoint=True,
            max_starting=(1, 7),
            cost=100
        )
        self.assertEqual(self.MACHINE, m_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        m_diff = InternalMachine(
            name='md',
            commodity='in_com',
            setpoint=self.SETPOINT,
            discrete_setpoint=False,
            max_starting=None,
            cost=0
        )
        self.assertNotEqual(self.MACHINE, m_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(self.MACHINE.key, 'm', msg="Wrong machine key name stored")
        self.assertEqual(self.MACHINE.kind, 'machine', msg="Machine node is not labelled as machine")
        self.assertTrue(self.MACHINE.commodity_in, msg="Machine node should have input commodities")
        self.assertTrue(self.MACHINE.commodity_out, msg="Machine node should have output commodities")
        self.assertSetEqual(self.MACHINE.commodities_in, {'in_com'}, msg="Wrong machine inputs stored")
        self.assertSetEqual(
            self.MACHINE.commodities_out,
            {'out_com_1', 'out_com_2'},
            msg="Wrong machine outputs stored"
        )
        self.assertListEqual(list(self.MACHINE.setpoint.index), [50.0, 75.0, 100.0], msg="Wrong setpoint index stored")

    def test_operations(self):
        # test discrete setpoint
        m = InternalMachine(
            name='m',
            commodity='in_com',
            setpoint=self.SETPOINT,
            discrete_setpoint=True,
            max_starting=None,
            cost=0
        )
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
        # test continuous setpoint
        m = InternalMachine(
            name='m',
            commodity='in_com',
            setpoint=self.SETPOINT,
            discrete_setpoint=False,
            max_starting=None,
            cost=0
        )
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

    def test_exposed(self):
        m = self.MACHINE.exposed
        self.assertIsInstance(m, Machine, msg="Wrong exposed type")
        # test stored information
        self.assertEqual(m.name, 'm', msg="Wrong exposed name")
        self.assertSetEqual(m.commodities_in, {'in_com'}, msg="Wrong exposed inputs")
        self.assertSetEqual(m.commodities_out, {'out_com_1', 'out_com_2'}, msg="Wrong exposed outputs")
        self.assertDictEqual(m.setpoint.to_dict(), self.SETPOINT.to_dict(), msg='Wrong exposed setpoint')
        self.assertEqual(m.cost, 0.0, msg='Wrong exposed cost')
        self.assertIsNone(m.max_starting, msg='Wrong exposed max starting')
        self.assertFalse(m.discrete_setpoint, msg='Wrong exposed discrete setpoint')
        # test immutability of mutable types
        m.setpoint.iloc[0, 0] = 5.0
        self.assertEqual(m.setpoint.iloc[0, 0], 5.0, msg="Exposed setpoint should be mutable")
        self.assertEqual(self.MACHINE.setpoint.iloc[0, 0], 0.0, msg="Internal setpoint should be immutable")
