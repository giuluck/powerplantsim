from ppsim.datatypes import InternalMachine, Machine
from test.datatypes.datatype import TestDataType, SETPOINT, HORIZON

MACHINE = InternalMachine(
    name='m',
    commodity='in_com',
    _setpoint=SETPOINT,
    discrete_setpoint=False,
    max_starting=None,
    cost=0,
    _horizon=HORIZON
)

COST_EXCEPTION = lambda v: f"The operating cost of the machine must be non-negative, got {v}"
SETPOINT_EXCEPTION = lambda v: f"Setpoints should be non-negative, got {v}"
OUTPUT_FLOWS_EXCEPTION = lambda c, v: f"Output flows should be non-negative, got {c}: {v}"
DISCRETE_SETPOINT_EXCEPTION = lambda v: f"Unsupported flow {v} for discrete setpoint [50.0, 75.0, 100.0]"
CONTINUOUS_SETPOINT_EXCEPTION = lambda v: f"Unsupported flow {v} for continuous setpoint 50.0 <= flow <= 100.0"


class TestMachine(TestDataType):

    def test_inputs(self):
        # test incorrect cost
        with self.assertRaises(AssertionError, msg="Negative cost should raise exception") as e:
            InternalMachine(
                name='m',
                commodity='in_com',
                _setpoint=SETPOINT,
                discrete_setpoint=True,
                max_starting=None,
                cost=-1.0,
                _horizon=HORIZON
            )
        self.assertEqual(
            str(e.exception),
            COST_EXCEPTION(-1.0),
            msg='Wrong exception message returned for negative cost on machine'
        )
        # test incorrect index setpoint (null input flow should not raise exception anymore)
        sp2 = SETPOINT.copy()
        sp2.index = [50.0, 0.0, 100.0]
        InternalMachine(
            name='m',
            commodity='in_com',
            _setpoint=sp2,
            discrete_setpoint=False,
            max_starting=None,
            cost=0,
            _horizon=HORIZON
        )
        sp2.index = [50.0, -1.0, 100.0]
        with self.assertRaises(AssertionError, msg="Negative input flows in setpoint should raise exception") as e:
            InternalMachine(
                name='m',
                commodity='in_com',
                _setpoint=sp2,
                discrete_setpoint=False,
                max_starting=None,
                cost=0,
                _horizon=HORIZON
            )
        self.assertEqual(
            str(e.exception),
            SETPOINT_EXCEPTION(-1.0),
            msg='Wrong exception message returned for negative setpoint on machine'
        )
        # test incorrect output flows
        sp2 = SETPOINT.copy()
        sp2['out_com_1'] = [-1.0, 0.0, 0.5]
        with self.assertRaises(AssertionError, msg="Negative output flows in setpoint should raise exception") as e:
            InternalMachine(
                name='m',
                commodity='in_com',
                _setpoint=sp2,
                discrete_setpoint=False,
                max_starting=None,
                cost=0,
                _horizon=HORIZON
            )
        self.assertEqual(
            str(e.exception),
            OUTPUT_FLOWS_EXCEPTION('out_com_1', -1.0),
            msg='Wrong exception message returned for negative output flows on machine'
        )
        sp2 = SETPOINT.copy()
        sp2['out_com_2'] = [60.0, -10.0, 30.0]
        with self.assertRaises(AssertionError, msg="Negative output flows in setpoint should raise exception") as e:
            InternalMachine(
                name='m',
                commodity='in_com',
                _setpoint=sp2,
                discrete_setpoint=False,
                max_starting=None,
                cost=0,
                _horizon=HORIZON
            )
        self.assertEqual(
            str(e.exception),
            OUTPUT_FLOWS_EXCEPTION('out_com_2', -10.0),
            msg='Wrong exception message returned for negative output flows on machine'
        )

    def test_hashing(self):
        # test equal hash
        m_equal = InternalMachine(
            name='m',
            commodity='in_com_2',
            _setpoint=SETPOINT.copy(),
            discrete_setpoint=True,
            max_starting=(1, 7),
            cost=100,
            _horizon=HORIZON
        )
        self.assertEqual(MACHINE, m_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        m_diff = InternalMachine(
            name='md',
            commodity='in_com',
            _setpoint=SETPOINT,
            discrete_setpoint=False,
            max_starting=None,
            cost=0,
            _horizon=HORIZON
        )
        self.assertNotEqual(MACHINE, m_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(MACHINE.key, 'm', msg="Wrong machine key name stored")
        self.assertEqual(MACHINE.kind, 'machine', msg="Machine node is not labelled as machine")
        self.assertEqual(MACHINE.commodity_in, 'in_com', msg="Wrong machine inputs stored")
        self.assertSetEqual(
            MACHINE.commodities_out,
            {'out_com_1', 'out_com_2'},
            msg="Wrong machine outputs stored"
        )
        self.assertDictEqual(MACHINE.setpoint.to_dict(), SETPOINT.to_dict(), msg='Wrong setpoint stored')

    def test_exposed(self):
        m = MACHINE.exposed
        self.assertIsInstance(m, Machine, msg="Wrong exposed type")
        # test stored information
        self.assertEqual(m.name, 'm', msg="Wrong exposed name")
        self.assertEqual(m.commodity_in, 'in_com', msg="Wrong exposed inputs")
        self.assertSetEqual(m.commodities_out, {'out_com_1', 'out_com_2'}, msg="Wrong exposed outputs")
        self.assertDictEqual(m.setpoint.to_dict(), SETPOINT.to_dict(), msg='Wrong exposed setpoint')
        self.assertEqual(m.cost, 0.0, msg='Wrong exposed cost')
        self.assertIsNone(m.max_starting, msg='Wrong exposed max starting')
        self.assertFalse(m.discrete_setpoint, msg='Wrong exposed discrete setpoint')
        # test dict (setpoint need to be tested separately due to errors in the equality check)
        m_dict = m.dict
        m_set = m_dict.pop('setpoint')
        self.assertEqual(m_dict, {
            'name': 'm',
            'commodity_in': 'in_com',
            'commodities_out': {'out_com_1', 'out_com_2'},
            'discrete_setpoint': False,
            'max_starting': None,
            'cost': 0
        }, msg='Wrong dictionary returned for exposed machine')
        self.assertDictEqual(m_set.to_dict(), SETPOINT.to_dict(), msg='Wrong dictionary returned for exposed machine')
        # test immutability of mutable types
        m.setpoint.iloc[0, 0] = 5.0
        self.assertEqual(m.setpoint.iloc[0, 0], 5.0, msg="Exposed setpoint should be mutable")
        self.assertEqual(MACHINE.setpoint.iloc[0, 0], 0.0, msg="Internal setpoint should be immutable")

# def test_operation(self):
#     # test discrete setpoint
#     m = InternalMachine(
#         name='m',
#         commodity='in_com',
#         _setpoint=SETPOINT,
#         discrete_setpoint=True,
#         max_starting=None,
#         cost=0,
#         _horizon=HORIZON
#     )
#     self.assertIsNone(m.operate(0.0), msg="Operate function should return None if the input is null")
#     self.assertDictEqual(m.operate(50.0), {'out_com_1': 0.0, 'out_com_2': 10.0}, msg="Wrong operation output")
#     self.assertDictEqual(m.operate(75.0), {'out_com_1': 0.5, 'out_com_2': 30.0}, msg="Wrong operation output")
#     self.assertDictEqual(m.operate(100.0), {'out_com_1': 1.0, 'out_com_2': 60.0}, msg="Wrong operation output")
#     with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-data") as e:
#         m.operate(70.0)
#     self.assertEqual(
#         str(e.exception),
#         DISCRETE_SETPOINT_EXCEPTION(70.0),
#         msg='Wrong exception message returned for discrete setpoint operation on machine'
#     )
#     with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-data") as e:
#         m.operate(90.0)
#     self.assertEqual(
#         str(e.exception),
#         DISCRETE_SETPOINT_EXCEPTION(90.0),
#         msg='Wrong exception message returned for discrete setpoint operation on machine'
#     )
#     with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-bound") as e:
#         m.operate(20.0)
#     self.assertEqual(
#         str(e.exception),
#         DISCRETE_SETPOINT_EXCEPTION(20.0),
#         msg='Wrong exception message returned for discrete setpoint operation on machine'
#     )
#     with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-bound") as e:
#         m.operate(110.0)
#     self.assertEqual(
#         str(e.exception),
#         DISCRETE_SETPOINT_EXCEPTION(110.0),
#         msg='Wrong exception message returned for discrete setpoint operation on machine'
#     )
#     # test continuous setpoint
#     m = InternalMachine(
#         name='m',
#         commodity='in_com',
#         _setpoint=SETPOINT,
#         discrete_setpoint=False,
#         max_starting=None,
#         cost=0,
#         _horizon=HORIZON
#     )
#     self.assertIsNone(m.operate(0.0), msg="Operate function should return None if the input is null")
#     self.assertDictEqual(m.operate(50.0), {'out_com_1': 0.0, 'out_com_2': 10.0}, msg="Wrong operation output")
#     self.assertDictEqual(m.operate(75.0), {'out_com_1': 0.5, 'out_com_2': 30.0}, msg="Wrong operation output")
#     self.assertDictEqual(m.operate(100.0), {'out_com_1': 1.0, 'out_com_2': 60.0}, msg="Wrong operation output")
#     self.assertDictEqual(m.operate(70.0), {'out_com_1': 0.4, 'out_com_2': 26.0}, msg="Wrong operation output")
#     self.assertDictEqual(m.operate(90.0), {'out_com_1': 0.8, 'out_com_2': 48.0}, msg="Wrong operation output")
#     with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-bound") as e:
#         m.operate(20.0)
#     self.assertEqual(
#         str(e.exception),
#         CONTINUOUS_SETPOINT_EXCEPTION(20.0),
#         msg='Wrong exception message returned for continuous setpoint operation on machine'
#     )
#     with self.assertRaises(AssertionError, msg="Discrete setpoint should not return values for out-of-bound") as e:
#         m.operate(110.0)
#     self.assertEqual(
#         str(e.exception),
#         CONTINUOUS_SETPOINT_EXCEPTION(110.0),
#         msg='Wrong exception message returned for continuous setpoint operation on machine'
#     )
