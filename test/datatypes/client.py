from ppsim.datatypes import InternalClient, Client
from test.datatypes.datatype import TestDataType


class TestClient(TestDataType):
    CLIENT = InternalClient(
        name='c',
        commodity='c_com',
        demands=TestDataType.SERIES_1,
        _variance=TestDataType.VARIANCE_1
    )

    def test_checks(self):
        pass

    def test_hashing(self):
        # test equal hash
        c_equal = InternalClient(name='c', commodity='c_com_2', demands=self.SERIES_2, _variance=self.VARIANCE_2)
        self.assertEqual(self.CLIENT, c_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        c_diff = InternalClient(name='cd', commodity='c_com', demands=self.SERIES_1, _variance=self.VARIANCE_1)
        self.assertNotEqual(self.CLIENT, c_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(self.CLIENT.key, 'c', msg="Wrong client key name stored")
        self.assertEqual(self.CLIENT.kind, 'client', msg="Client node is not labelled as client")
        self.assertTrue(self.CLIENT.commodity_in, msg="Client node should have input commodities")
        self.assertFalse(self.CLIENT.commodity_out, msg="Client node should not have output commodities")
        self.assertSetEqual(self.CLIENT.commodities_in, {'c_com'}, msg="Wrong client inputs stored")
        self.assertSetEqual(self.CLIENT.commodities_out, set(), msg="Wrong client outputs stored")

    def test_operations(self):
        pass

    def test_exposed(self):
        c = self.CLIENT.exposed
        self.assertIsInstance(c, Client, msg="Wrong exposed type")
        # test stored information
        self.assertEqual(c.name, 'c', msg="Wrong exposed name")
        self.assertSetEqual(c.commodities_in, {'c_com'}, msg="Wrong exposed inputs")
        self.assertSetEqual(c.commodities_out, set(), msg="Wrong exposed outputs")
        self.assertListEqual(list(c.demands), list(self.SERIES_1), msg='Wrong exposed demands')
        # test immutability of mutable types
        c.demands[0] = 5.0
        self.assertEqual(c.demands[0], 5.0, msg="Exposed demands should be mutable")
        self.assertEqual(self.CLIENT.demands[0], 3.0, msg="Internal demands should be immutable")
