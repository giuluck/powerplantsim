from ppsim.datatypes import InternalSupplier, Supplier
from test.datatypes.datatype import TestDataType


class TestSupplier(TestDataType):
    SUPPLIER = InternalSupplier(
        name='s',
        commodity='s_com',
        prices=TestDataType.SERIES_1,
        variance_fn=TestDataType.VARIANCE_1
    )

    def test_checks(self):
        pass

    def test_hashing(self):
        # test equal hash
        s_equal = InternalSupplier(name='s', commodity='s_com_2', prices=self.SERIES_2, variance_fn=self.VARIANCE_2)
        self.assertEqual(self.SUPPLIER, s_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        s_diff = InternalSupplier(name='sd', commodity='s_com', prices=self.SERIES_1, variance_fn=self.VARIANCE_1)
        self.assertNotEqual(self.SUPPLIER, s_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(self.SUPPLIER.key, 's', msg="Wrong supplier key name stored")
        self.assertEqual(self.SUPPLIER.kind, 'supplier', msg="Supplier node is not labelled as supplier")
        self.assertIsNone(self.SUPPLIER.commodity_in, msg="Wrong supplier inputs stored")
        self.assertSetEqual(self.SUPPLIER.commodities_out, {'s_com'}, msg="Wrong supplier outputs stored")

    def test_operations(self):
        pass

    def test_exposed(self):
        s = self.SUPPLIER.exposed
        self.assertIsInstance(s, Supplier, msg="Wrong exposed type")
        # test stored information
        self.assertEqual(s.name, 's', msg="Wrong exposed name")
        self.assertIsNone(s.commodity_in, msg="Wrong exposed inputs")
        self.assertSetEqual(s.commodities_out, {'s_com'}, msg="Wrong exposed outputs")
        self.assertListEqual(list(s.prices), list(self.SERIES_1), msg='Wrong exposed demands')
        # test immutability of mutable types
        s.prices[0] = 5.0
        self.assertEqual(s.prices[0], 5.0, msg="Exposed prices should be mutable")
        self.assertEqual(self.SUPPLIER.prices[0], 3.0, msg="Internal prices should be immutable")
