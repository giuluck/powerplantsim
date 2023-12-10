from ppsim.datatypes import InternalStorage, Storage
from test.datatypes.datatype import TestDataType


class TestStorage(TestDataType):
    STORAGE = InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=1.0)

    def test_checks(self):
        # test correct dissipation
        InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=0.0)
        InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=0.5)
        InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=1.0)
        # test incorrect dissipation
        with self.assertRaises(AssertionError, msg="Out of bound dissipation should raise exception"):
            InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=-1.0)
        with self.assertRaises(AssertionError, msg="Out of bound dissipation should raise exception"):
            InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=2.0)
        # test incorrect capacity
        with self.assertRaises(AssertionError, msg="Null capacity should raise exception"):
            InternalStorage(name='s', commodity='s_com', capacity=0, dissipation=1.0)
        with self.assertRaises(AssertionError, msg="Negative capacity should raise exception"):
            InternalStorage(name='s', commodity='s_com', capacity=-10, dissipation=1.0)

    def test_hashing(self):
        # test equal hash
        s_equal = InternalStorage(name='s', commodity='s_com_2', capacity=50, dissipation=1.0)
        self.assertEqual(self.STORAGE, s_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        s_diff = InternalStorage(name='sd', commodity='s_com', capacity=100, dissipation=1.0)
        self.assertNotEqual(self.STORAGE, s_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(self.STORAGE.key, 's', msg="Wrong storage key name stored")
        self.assertEqual(self.STORAGE.kind, 'storage', msg="Storage node is not labelled as supplier")
        self.assertTrue(self.STORAGE.commodity_in, msg="Storage node should have input commodities")
        self.assertTrue(self.STORAGE.commodity_out, msg="Storage node should have output commodities")
        self.assertSetEqual(self.STORAGE.commodities_in, {'s_com'}, msg="Wrong storage inputs stored")
        self.assertSetEqual(self.STORAGE.commodities_out, {'s_com'}, msg="Wrong storage outputs stored")

    def test_operations(self):
        pass

    def test_exposed(self):
        s = self.STORAGE.exposed
        self.assertIsInstance(s, Storage, msg="Wrong exposed type")
        # test stored information
        self.assertEqual(s.name, 's', msg="Wrong exposed name")
        self.assertSetEqual(s.commodities_in, {'s_com'}, msg="Wrong exposed inputs")
        self.assertSetEqual(s.commodities_out, {'s_com'}, msg="Wrong exposed outputs")
        self.assertEqual(s.capacity, 100.0, msg='Wrong exposed capacity')
        self.assertEqual(s.dissipation, 1.0, msg='Wrong exposed dissipation')
