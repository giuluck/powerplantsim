import numpy as np

from ppsim.datatypes import Supplier
from test.datatypes.datatype import TestDataType, SERIES_1, SERIES_2, VARIANCE_1, VARIANCE_2, PLANT

SUPPLIER = Supplier(
    name='s',
    commodity='s_com',
    _predictions=SERIES_1,
    _variance_fn=VARIANCE_1,
    _plant=PLANT
)


class TestSupplier(TestDataType):

    def test_inputs(self):
        pass

    def test_hashing(self):
        # test equal hash
        s_equal = Supplier(name='s', commodity='s_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2, _plant=PLANT)
        self.assertEqual(SUPPLIER, s_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        s_diff = Supplier(name='sd', commodity='s_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1, _plant=PLANT)
        self.assertNotEqual(SUPPLIER, s_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(SUPPLIER.key, 's', msg="Wrong supplier key name stored")
        self.assertEqual(SUPPLIER.kind, 'supplier', msg="Supplier node is not labelled as supplier")
        self.assertSetEqual(SUPPLIER.commodities_in, set(), msg="Wrong supplier inputs stored")
        self.assertSetEqual(SUPPLIER.commodities_out, {'s_com'}, msg="Wrong supplier outputs stored")

    def test_immutability(self):
        SUPPLIER.prices[0] = 5.0
        self.assertEqual(len(SUPPLIER.prices), 0, msg="Supplier prices should be immutable")

    def test_dict(self):
        # pandas series need to be tested separately due to errors in the equality check
        s_dict = SUPPLIER.dict
        s_val = s_dict.pop('prices')
        self.assertEqual(s_dict, {
            'name': 's',
            'kind': 'supplier',
            'commodity': 's_com',
            'current_price': None,
        }, msg='Wrong dictionary returned for supplier')
        self.assertDictEqual(s_val.to_dict(), {}, msg='Wrong dictionary returned for supplier')

    def test_operation(self):
        s = SUPPLIER.copy()
        rng = np.random.default_rng(0)
        val = 3.0 + np.random.default_rng(0).normal()
        self.assertDictEqual(s.prices.to_dict(), {}, msg=f"Supplier prices should be empty before the simulation")
        self.assertIsNone(s.current_price, msg=f"Supplier current price should be None outside of the simulation")
        s.update(rng=rng, flows={}, states={})
        self.assertDictEqual(s.prices.to_dict(), {}, msg=f"Supplier prices should be empty before step")
        self.assertEqual(s.current_price, val, msg=f"Supplier current price should be stored after update")
        s.step(flows={}, states={})
        self.assertDictEqual(s.prices.to_dict(), {0: val}, msg=f"Supplier prices should be filled after step")
        self.assertIsNone(s.current_price, msg=f"Supplier current price should be None outside of the simulation")
