from ppsim.datatypes import InternalStorage, InternalEdge, InternalMachine, Edge, InternalSupplier
from test.datatypes.datatype import TestDataType, SETPOINT, SERIES_1, VARIANCE_1

MACHINE = InternalMachine(
    name='m',
    commodity='in_com',
    setpoint=SETPOINT,
    discrete_setpoint=False,
    max_starting=None,
    cost=0
)

SUPPLIER = InternalSupplier(
    name='s',
    commodity='in_com',
    prices=SERIES_1,
    variance_fn=VARIANCE_1
)

STORAGE_1 = InternalStorage(name='s1', commodity='out_com_1', capacity=100, dissipation=1.0)

STORAGE_2 = InternalStorage(name='s2', commodity='out_com_2', capacity=100, dissipation=1.0)

EDGE = InternalEdge(
    source=MACHINE,
    destination=STORAGE_1,
    min_flow=0.0,
    max_flow=100.0,
    integer=False
)

MIN_FLOW_EXCEPTION = lambda v: f"The minimum flow cannot be negative, got {v}"
MAX_FLOW_EXCEPTION = lambda v_min, v_max: f"The maximum flow cannot be lower than the minimum, got {v_max} < {v_min}"
EMPTY_DESTINATION_EXCEPTION = lambda v: f"Destination node {v} does not accept any input commodity, but it should"
INCONSISTENT_SOURCE_EXCEPTION = lambda c, s: f"Source node should return {c}, but it returns {s} only"


class TestEdge(TestDataType):

    def test_inputs(self):
        # check correct flows
        InternalEdge(source=MACHINE, destination=STORAGE_1, min_flow=0.0, max_flow=1.0, integer=False)
        InternalEdge(source=MACHINE, destination=STORAGE_1, min_flow=1.0, max_flow=2.0, integer=False)
        InternalEdge(source=MACHINE, destination=STORAGE_1, min_flow=1.0, max_flow=1.0, integer=False)
        # check incorrect flows
        with self.assertRaises(AssertionError, msg="Negative min flow should raise exception") as e:
            InternalEdge(source=MACHINE, destination=STORAGE_1, min_flow=-1.0, max_flow=100.0, integer=False)
        self.assertEqual(
            str(e.exception),
            MIN_FLOW_EXCEPTION(-1.0),
            msg='Wrong exception message returned for negative min flow on edge'
        )
        with self.assertRaises(AssertionError, msg="max flow < min flow should raise exception") as e:
            InternalEdge(source=MACHINE, destination=STORAGE_1, min_flow=101.0, max_flow=100.0, integer=False)
        self.assertEqual(
            str(e.exception),
            MAX_FLOW_EXCEPTION(101.0, 100.0),
            msg='Wrong exception message returned for max flow < min flow on edge'
        )
        # check incorrect commodities
        with self.assertRaises(AssertionError, msg="Empty destination commodity should raise exception") as e:
            InternalEdge(source=MACHINE, destination=SUPPLIER, min_flow=0.0, max_flow=100.0, integer=False)
        self.assertEqual(
            str(e.exception),
            EMPTY_DESTINATION_EXCEPTION('s'),
            msg='Wrong exception message returned for empty destination commodity on edge'
        )
        with self.assertRaises(AssertionError, msg="Wrong source commodity should raise exception") as e:
            InternalEdge(source=STORAGE_1, destination=MACHINE, min_flow=0.0, max_flow=100.0, integer=False)
        self.assertEqual(
            str(e.exception),
            INCONSISTENT_SOURCE_EXCEPTION('in_com', {'out_com_1'}),
            msg='Wrong exception message returned for wrong source commodity on edge'
        )

    def test_hashing(self):
        # test equal hash
        e_equal = InternalEdge(
            source=MACHINE,
            destination=STORAGE_1,
            min_flow=50.0,
            max_flow=60.0,
            integer=True
        )
        self.assertEqual(EDGE, e_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        e_diff = InternalEdge(
            source=MACHINE,
            destination=STORAGE_2,
            min_flow=0.0,
            max_flow=100.0,
            integer=False
        )
        self.assertNotEqual(EDGE, e_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertIsInstance(EDGE.key, InternalEdge.EdgeID, msg="Wrong edge key type stored")
        self.assertTupleEqual(tuple(EDGE.key), ('m', 's1', 'out_com_1'), msg="Wrong edge key stored")

    def test_exposed(self):
        e = EDGE.exposed
        self.assertIsInstance(e, Edge, msg="Wrong exposed type")
        # test stored information
        self.assertEqual(e.source.name, 'm', msg="Wrong exposed source")
        self.assertEqual(e.destination.name, 's1', msg="Wrong exposed destination")
        self.assertEqual(e.commodity, 'out_com_1', msg="Wrong exposed commodity")
        self.assertEqual(e.min_flow, 0.0, msg='Wrong exposed min flow')
        self.assertEqual(e.max_flow, 100.0, msg='Wrong exposed max flow')
        self.assertFalse(e.integer, msg='Wrong exposed integer')
