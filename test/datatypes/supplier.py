from ppsim.datatypes import Supplier
from test.datatypes.datatype import TestDataType, SERIES_1, SERIES_2, VARIANCE_1, VARIANCE_2

SUPPLIER = Supplier(
    name='s',
    commodity='s_com',
    _predictions=SERIES_1,
    _variance_fn=VARIANCE_1
)


class TestSupplier(TestDataType):

    def test_inputs(self):
        pass

    def test_hashing(self):
        # test equal hash
        s_equal = Supplier(name='s', commodity='s_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2)
        self.assertEqual(SUPPLIER, s_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        s_diff = Supplier(name='sd', commodity='s_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1)
        self.assertNotEqual(SUPPLIER, s_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(SUPPLIER.key, 's', msg="Wrong supplier key name stored")
        self.assertEqual(SUPPLIER.kind, 'supplier', msg="Supplier node is not labelled as supplier")
        self.assertIsNone(SUPPLIER.commodity_in, msg="Wrong supplier inputs stored")
        self.assertSetEqual(SUPPLIER.commodities_out, {'s_com'}, msg="Wrong supplier outputs stored")

    def test_immutability(self):
        SUPPLIER.prices[0] = 5.0
        SUPPLIER.predicted_prices[0] = 5.0
        self.assertEqual(len(SUPPLIER.prices), 0, msg="Supplier prices should be immutable")
        self.assertEqual(SUPPLIER.predicted_prices[0], 3.0, msg="Supplier predicted prices should be immutable")

    def test_dict(self):
        # pandas series need to be tested separately due to errors in the equality check
        s_dict = SUPPLIER.dict
        s_val = s_dict.pop('prices')
        s_pred = s_dict.pop('predicted_prices')
        self.assertEqual(s_dict, {
            'name': 's',
            'commodity_in': None,
            'commodities_out': {'s_com'},
        }, msg='Wrong dictionary returned for supplier')
        self.assertDictEqual(s_val.to_dict(), {}, msg='Wrong dictionary returned for supplier')
        self.assertDictEqual(s_pred.to_dict(), SERIES_1.to_dict(), msg='Wrong dictionary returned for supplier')
