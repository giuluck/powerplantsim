from ppsim.datatypes import InternalClient, Client
from test.datatypes.datatype import TestDataType, SERIES_1, SERIES_2, VARIANCE_1, VARIANCE_2

CLIENT = InternalClient(
    name='c',
    commodity='c_com',
    demands=SERIES_1,
    variance_fn=VARIANCE_1
)


class TestClient(TestDataType):

    def test_inputs(self):
        pass

    def test_hashing(self):
        # test equal hash
        c_equal = InternalClient(name='c', commodity='c_com_2', demands=SERIES_2, variance_fn=VARIANCE_2)
        self.assertEqual(CLIENT, c_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        c_diff = InternalClient(name='cd', commodity='c_com', demands=SERIES_1, variance_fn=VARIANCE_1)
        self.assertNotEqual(CLIENT, c_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(CLIENT.key, 'c', msg="Wrong client key name stored")
        self.assertEqual(CLIENT.kind, 'client', msg="Client node is not labelled as client")
        self.assertEqual(CLIENT.commodity_in, 'c_com', msg="Wrong client inputs stored")
        self.assertSetEqual(CLIENT.commodities_out, set(), msg="Wrong client outputs stored")

    def test_exposed(self):
        c = CLIENT.exposed
        self.assertIsInstance(c, Client, msg="Wrong exposed type")
        # test stored information
        self.assertEqual(c.name, 'c', msg="Wrong exposed name")
        self.assertEqual(c.commodity_in, 'c_com', msg="Wrong exposed inputs")
        self.assertSetEqual(c.commodities_out, set(), msg="Wrong exposed outputs")
        self.assertListEqual(list(c.demands), list(SERIES_1), msg='Wrong exposed demands')
        # test immutability of mutable types
        c.demands[0] = 5.0
        self.assertEqual(c.demands[0], 5.0, msg="Exposed demands should be mutable")
        self.assertEqual(CLIENT.demands[0], 3.0, msg="Internal demands should be immutable")
