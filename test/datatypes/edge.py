from ppsim.datatypes import Supplier, Storage, Machine, Edge
from test.datatypes.datatype import TestDataType, SETPOINT, SERIES_1, VARIANCE_1, PLANT

MACHINE = Machine(
    name='m',
    commodity='in_com',
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

EDGE = Edge(
    _source=MACHINE,
    _destination=STORAGE_1,
    commodity='out_com_1',
    min_flow=0.0,
    max_flow=100.0,
    integer=False,
    _plant=PLANT
)

MIN_FLOW_EXCEPTION = lambda v: f"The minimum flow cannot be negative, got {v}"
MAX_FLOW_EXCEPTION = lambda v_min, v_max: f"The maximum flow cannot be lower than the minimum, got {v_max} < {v_min}"
EMPTY_DESTINATION_EXCEPTION = lambda v: f"Destination node '{v}' does not accept any input commodity, but it should"
INCONSISTENT_SOURCE_EXCEPTION = lambda n, c, s: f"Source node '{n}' should return commodity '{c}', but it returns {s}"
INCONSISTENT_DESTINATION_EXCEPTION = \
    lambda n, c, s: f"Destination node '{n}' should accept commodity '{c}', but it accepts {s}"


class TestEdge(TestDataType):

    def test_inputs(self):
        # check correct flows
        Edge(
            _source=MACHINE,
            _destination=STORAGE_1,
            commodity='out_com_1',
            min_flow=0.0,
            max_flow=1.0,
            integer=False,
            _plant=PLANT
        )
        Edge(
            _source=MACHINE,
            _destination=STORAGE_1,
            commodity='out_com_1',
            min_flow=1.0,
            max_flow=2.0,
            integer=False,
            _plant=PLANT
        )
        Edge(
            _source=MACHINE,
            _destination=STORAGE_1,
            commodity='out_com_1',
            min_flow=1.0,
            max_flow=1.0,
            integer=False,
            _plant=PLANT
        )
        # check incorrect flows
        with self.assertRaises(AssertionError, msg="Negative min flow should raise exception") as e:
            Edge(
                _source=MACHINE,
                _destination=STORAGE_1,
                commodity='out',
                min_flow=-1.0,
                max_flow=100.0,
                integer=False,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            MIN_FLOW_EXCEPTION(-1.0),
            msg='Wrong exception message returned for negative min flow on edge'
        )
        with self.assertRaises(AssertionError, msg="max flow < min flow should raise exception") as e:
            Edge(
                _source=MACHINE,
                _destination=STORAGE_1,
                commodity='out_com_1',
                min_flow=101.0,
                max_flow=100.0,
                integer=False,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            MAX_FLOW_EXCEPTION(101.0, 100.0),
            msg='Wrong exception message returned for max flow < min flow on edge'
        )
        # check incorrect commodities
        with self.assertRaises(AssertionError, msg="Empty destination commodity should raise exception") as e:
            Edge(
                _source=MACHINE,
                _destination=SUPPLIER,
                commodity='out_com_1',
                min_flow=0.0,
                max_flow=100.0,
                integer=False,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            EMPTY_DESTINATION_EXCEPTION('s'),
            msg='Wrong exception message returned for empty destination commodity on edge'
        )
        with self.assertRaises(AssertionError, msg="Wrong source commodity should raise exception") as e:
            Edge(
                _source=STORAGE_1,
                _destination=MACHINE,
                commodity='in_com',
                min_flow=0.0,
                max_flow=100.0,
                integer=False,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            INCONSISTENT_SOURCE_EXCEPTION('s1', 'in_com', {'out_com_1'}),
            msg='Wrong exception message returned for wrong source commodity on edge'
        )
        with self.assertRaises(AssertionError, msg="Wrong destination commodity should raise exception") as e:
            Edge(
                _source=STORAGE_1,
                _destination=MACHINE,
                commodity='out_com_1',
                min_flow=0.0,
                max_flow=100.0,
                integer=False,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            INCONSISTENT_DESTINATION_EXCEPTION('m', 'out_com_1', {'in_com'}),
            msg='Wrong exception message returned for wrong source commodity on edge'
        )

    def test_hashing(self):
        # test equal hash
        e_equal = Edge(
            _source=MACHINE,
            _destination=STORAGE_1,
            commodity='out_com_1',
            min_flow=50.0,
            max_flow=60.0,
            integer=True,
            _plant=PLANT
        )
        self.assertEqual(EDGE, e_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        e_diff = Edge(
            _source=MACHINE,
            _destination=STORAGE_2,
            commodity='out_com_2',
            min_flow=0.0,
            max_flow=100.0,
            integer=False,
            _plant=PLANT
        )
        self.assertNotEqual(EDGE, e_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertIsInstance(EDGE.key, tuple, msg="Wrong edge key type stored")
        self.assertTupleEqual(EDGE.key, ('m', 's1'), msg="Wrong edge key stored")

    def test_immutability(self):
        EDGE.flows[0] = 5.0
        self.assertEqual(len(EDGE.flows), 0, msg="Edge flows should be immutable")

    def test_dict(self):
        e_dict = EDGE.dict
        e_flows = e_dict.pop('flows')
        self.assertEqual(e_dict, {
            'source': 'm',
            'destination': 's1',
            'commodity': 'out_com_1',
            'min_flow': 0.0,
            'max_flow': 100.0,
            'integer': False
        }, msg='Wrong dictionary returned for edge')
        self.assertDictEqual(e_flows.to_dict(), {}, msg='Wrong dictionary returned for edge')
