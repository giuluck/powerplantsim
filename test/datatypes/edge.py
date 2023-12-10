from ppsim.datatypes import InternalStorage, InternalEdge, InternalMachine, Edge
from test.datatypes.datatype import TestDataType


class TestEdge(TestDataType):
    MACHINE = InternalMachine(
        name='m',
        commodity='in_com',
        setpoint=TestDataType.SETPOINT,
        discrete_setpoint=False,
        max_starting=None,
        cost=0
    )

    STORAGE_1 = InternalStorage(name='s1', commodity='out_com_1', capacity=100, dissipation=1.0)

    STORAGE_2 = InternalStorage(name='s2', commodity='out_com_2', capacity=100, dissipation=1.0)

    EDGE = e = InternalEdge(
        source=MACHINE,
        destination=STORAGE_1,
        commodity='out_com_1',
        min_flow=0.0,
        max_flow=100.0,
        integer=False
    )

    def test_checks(self):
        m, s = self.MACHINE, self.STORAGE_1
        # check correct flows
        InternalEdge(source=m, destination=s, commodity='out_com_1', min_flow=0.0, max_flow=1.0, integer=False)
        InternalEdge(source=m, destination=s, commodity='out_com_1', min_flow=1.0, max_flow=2.0, integer=False)
        InternalEdge(source=m, destination=s, commodity='out_com_1', min_flow=1.0, max_flow=1.0, integer=False)
        # check incorrect flows
        with self.assertRaises(AssertionError, msg="Negative min flow should raise exception"):
            InternalEdge(source=m, destination=s, commodity='out_com_1', min_flow=-1.0, max_flow=100.0, integer=False)
        with self.assertRaises(AssertionError, msg="max flow < min flow should raise exception"):
            InternalEdge(source=m, destination=s, commodity='out_com_1', min_flow=101.0, max_flow=100.0, integer=False)
        # check incorrect commodities
        with self.assertRaises(AssertionError, msg="Wrong source commodity should raise exception"):
            InternalEdge(source=s, destination=m, commodity='in_com', min_flow=0.0, max_flow=100.0, integer=False)
        with self.assertRaises(AssertionError, msg="Wrong destination commodity should raise exception"):
            InternalEdge(source=s, destination=m, commodity='out_com_1', min_flow=0.0, max_flow=100.0, integer=False)
        with self.assertRaises(AssertionError, msg="Wrong source and destination commodity should raise exception"):
            InternalEdge(source=m, destination=s, commodity='in_com', min_flow=0.0, max_flow=100.0, integer=False)

    def test_hashing(self):
        # test equal hash
        e_equal = InternalEdge(
            source=self.MACHINE,
            destination=self.STORAGE_1,
            commodity='out_com_1',
            min_flow=50.0,
            max_flow=60.0,
            integer=True
        )
        self.assertEqual(self.EDGE, e_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        e_diff = InternalEdge(
            source=self.MACHINE,
            destination=self.STORAGE_2,
            commodity='out_com_2',
            min_flow=0.0,
            max_flow=100.0,
            integer=False
        )
        self.assertNotEqual(self.EDGE, e_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertIsInstance(self.EDGE.key, InternalEdge.EdgeID, msg="Wrong edge key type stored")
        self.assertTupleEqual(tuple(self.EDGE.key), ('m', 's1', 'out_com_1'), msg="Wrong edge key stored")

    def test_operations(self):
        pass

    def test_exposed(self):
        e = self.EDGE.exposed
        self.assertIsInstance(e, Edge, msg="Wrong exposed type")
        # test stored information
        self.assertEqual(e.source.name, 'm', msg="Wrong exposed source")
        self.assertEqual(e.destination.name, 's1', msg="Wrong exposed destination")
        self.assertEqual(e.commodity, 'out_com_1', msg="Wrong exposed commodity")
        self.assertEqual(e.min_flow, 0.0, msg='Wrong exposed min flow')
        self.assertEqual(e.max_flow, 100.0, msg='Wrong exposed max flow')
        self.assertFalse(e.integer, msg='Wrong exposed integer')
