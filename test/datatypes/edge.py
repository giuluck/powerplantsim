import pyomo.environ as pyo

from ppsim.datatypes import Supplier, Storage, Machine, SingleEdge, MultiEdge
from test.datatypes.datatype import TestDataType, SETPOINT, SERIES_1, VARIANCE_1, PLANT

MACHINE = Machine(
    name='m',
    _setpoint=SETPOINT,
    discrete_setpoint=False,
    max_starting=None,
    cost=0.0,
    _plant=PLANT
)

SUPPLIER = Supplier(
    name='s',
    commodity='in_com',
    _predictions=SERIES_1,
    _variance_fn=VARIANCE_1,
    _plant=PLANT
)

STORAGE_1 = Storage(
    name='s1',
    commodity='out_com_1',
    capacity=100.0,
    dissipation=1.0,
    charge_rate=10.0,
    discharge_rate=10.0,
    _plant=PLANT
)

STORAGE_2 = Storage(
    name='s2',
    commodity='out_com_2',
    capacity=100.0,
    dissipation=1.0,
    charge_rate=10.0,
    discharge_rate=10.0,
    _plant=PLANT
)

SINGLE_EDGE = SingleEdge(
    _source=MACHINE,
    _destination=STORAGE_1,
    commodity='out_com_1',
    min_flow=0.0,
    max_flow=100.0,
    _plant=PLANT
)

MULTI_EDGE = MultiEdge(
    _source=MACHINE,
    _destination=STORAGE_1,
    commodity='out_com_1',
    min_flow=0.0,
    max_flow=100.0,
    _plant=PLANT
)

MIN_FLOW_EXCEPTION = lambda v: f"The minimum flow cannot be negative, got {v}"
MAX_FLOW_EXCEPTION = lambda v_min, v_max: f"The maximum flow cannot be lower than the minimum, got {v_max} < {v_min}"
INCONSISTENT_SOURCE_EXCEPTION = lambda n, c, s: f"Source node '{n}' should return commodity '{c}', but it returns {s}"
INCONSISTENT_DESTINATION_EXCEPTION = \
    lambda n, c, s: f"Destination node '{n}' should accept commodity '{c}', but it accepts {s}"
RECEIVED_MIN_FLOW_EXCEPTION = lambda e, m, f: f"Flow for edge {e} should be >= {m}, got {f}"
RECEIVED_MAX_FLOW_EXCEPTION = lambda e, m, f: f"Flow for edge {e} should be <= {m}, got {f}"


class TestEdge(TestDataType):

    def test_inputs(self):
        # check correct flows
        SingleEdge(
            _source=MACHINE,
            _destination=STORAGE_1,
            commodity='out_com_1',
            min_flow=0.0,
            max_flow=1.0,
            _plant=PLANT
        )
        SingleEdge(
            _source=MACHINE,
            _destination=STORAGE_1,
            commodity='out_com_1',
            min_flow=1.0,
            max_flow=2.0,
            _plant=PLANT
        )
        SingleEdge(
            _source=MACHINE,
            _destination=STORAGE_1,
            commodity='out_com_1',
            min_flow=1.0,
            max_flow=1.0,
            _plant=PLANT
        )
        # check incorrect flows
        with self.assertRaises(AssertionError, msg="Negative min flow should raise exception") as e:
            SingleEdge(
                _source=MACHINE,
                _destination=STORAGE_1,
                commodity='out',
                min_flow=-1.0,
                max_flow=100.0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            MIN_FLOW_EXCEPTION(-1.0),
            msg='Wrong exception message returned for negative min flow on edge'
        )
        with self.assertRaises(AssertionError, msg="max flow < min flow should raise exception") as e:
            SingleEdge(
                _source=MACHINE,
                _destination=STORAGE_1,
                commodity='out_com_1',
                min_flow=101.0,
                max_flow=100.0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            MAX_FLOW_EXCEPTION(101.0, 100.0),
            msg='Wrong exception message returned for max flow < min flow on edge'
        )
        # check incorrect commodities
        with self.assertRaises(AssertionError, msg="Wrong source commodity should raise exception") as e:
            SingleEdge(
                _source=STORAGE_1,
                _destination=MACHINE,
                commodity='in_com',
                min_flow=0.0,
                max_flow=100.0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            INCONSISTENT_SOURCE_EXCEPTION('s1', 'in_com', {'out_com_1'}),
            msg='Wrong exception message returned for wrong source commodity on edge'
        )
        with self.assertRaises(AssertionError, msg="Wrong destination commodity should raise exception") as e:
            SingleEdge(
                _source=STORAGE_1,
                _destination=MACHINE,
                commodity='out_com_1',
                min_flow=0.0,
                max_flow=100.0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            INCONSISTENT_DESTINATION_EXCEPTION('m', 'out_com_1', {'in_com'}),
            msg='Wrong exception message returned for wrong source commodity on edge'
        )

    def test_hashing(self):
        # test equal hash
        e_equal = SingleEdge(
            _source=MACHINE,
            _destination=STORAGE_1,
            commodity='out_com_1',
            min_flow=50.0,
            max_flow=60.0,
            _plant=PLANT
        )
        self.assertEqual(SINGLE_EDGE, e_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        e_diff = SingleEdge(
            _source=MACHINE,
            _destination=STORAGE_2,
            commodity='out_com_2',
            min_flow=0.0,
            max_flow=100.0,
            _plant=PLANT
        )
        self.assertNotEqual(SINGLE_EDGE, e_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        # test single edge
        self.assertEqual(SINGLE_EDGE.name, 'm --> s1', msg="Wrong simple edge name stored")
        self.assertIsInstance(SINGLE_EDGE.key, tuple, msg="Wrong simple edge key type stored")
        self.assertTupleEqual(SINGLE_EDGE.key, ('m', 's1'), msg="Wrong simple edge key stored")
        # test multi edge
        self.assertEqual(MULTI_EDGE.name, 'm --(out_com_1)--> s1', msg="Wrong multi edge name stored")
        self.assertIsInstance(MULTI_EDGE.key, tuple, msg="Wrong multi edge key type stored")
        self.assertTupleEqual(MULTI_EDGE.key, ('m', 's1', 'out_com_1'), msg="Wrong multi edge key stored")

    def test_immutability(self):
        SINGLE_EDGE.flows[0] = 5.0
        self.assertEqual(len(SINGLE_EDGE.flows), 0, msg="Edge flows should be immutable")

    def test_dict(self):
        e_dict = SINGLE_EDGE.dict
        self.assertDictEqual(e_dict, {
            'name': 'm --> s1',
            'source': 'm',
            'destination': 's1',
            'commodity': 'out_com_1',
            'min_flow': 0.0,
            'max_flow': 100.0,
            'bounds': (0.0, 100.0),
            'current_flow': None
        }, msg='Wrong dictionary returned for edge')

    def test_operation(self):
        e = SINGLE_EDGE.copy()
        self.assertDictEqual(e.flows.to_dict(), {}, msg=f"Edge flows should be empty before the simulation")
        self.assertIsNone(e.current_flow, msg=f"Edge current flow should be None outside of the simulation")
        e.update(rng=None, flows={e: 1.0}, states={})
        self.assertDictEqual(e.flows.to_dict(), {}, msg=f"Edge flows should be empty before step")
        self.assertAlmostEqual(e.current_flow, 1.0, msg=f"Edge current flow should be stored after update")
        e.step(flows={e: 0.0}, states={})
        self.assertDictEqual(e.flows.to_dict(), {0: 0.0}, msg=f"Edge flows should be filled after step")
        self.assertIsNone(e.current_flow, msg=f"Edge flow should be None outside of the simulation")
        # test min flow exception
        flows = {e: -1.0}
        with self.assertRaises(AssertionError, msg="Under bound received flow should raise exception") as x:
            e.step(flows=flows, states={})
        self.assertEqual(
            str(x.exception),
            RECEIVED_MIN_FLOW_EXCEPTION(('m', 's1'), 0.0, -1.0),
            msg='Wrong exception message returned for under bound received flow on edge'
        )
        # test max flow exception
        flows = {e: 101.0}
        e = SINGLE_EDGE.copy()
        e.update(rng=None, flows=flows, states={})
        with self.assertRaises(AssertionError, msg="Over bound received flow should raise exception") as x:
            e.step(flows=flows, states={})
        self.assertEqual(
            str(x.exception),
            RECEIVED_MAX_FLOW_EXCEPTION(('m', 's1'), 100.0, 101.0),
            msg='Wrong exception message returned for over bound received flow on edge'
        )

    def test_pyomo(self):
        e = SINGLE_EDGE.copy()
        e.update(rng=None, flows={e: 30.0}, states={})
        model = e.to_pyomo(mutable=False)
        self.assertIsInstance(model.flow, pyo.Var, msg="Wrong type variable for edge flow")
        self.assertEqual(model.flow.domain, pyo.NonNegativeReals, msg="Wrong variable domain stored for edge flow.")
        self.assertEqual(model.flow.bounds, (0.0, 100.0), msg="Wrong variable bounds stored for edge flow.")
        self.assertAlmostEqual(model.flow.value, 30.0, msg="Edge flow should be initialized to current flow")
        self.assertIsNone(e.to_pyomo(mutable=True).flow.value, msg="Edge flow should not be initialized")
