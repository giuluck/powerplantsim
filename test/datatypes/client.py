from ppsim.datatypes import InternalCustomer, InternalPurchaser, Customer, Purchaser
from test.datatypes.datatype import TestDataType, SERIES_1, SERIES_2, VARIANCE_1, VARIANCE_2

CUSTOMER = InternalCustomer(
    name='c',
    commodity='c_com',
    _predictions=SERIES_1,
    _variance_fn=VARIANCE_1
)

PURCHASER = InternalPurchaser(
    name='p',
    commodity='p_com',
    _predictions=SERIES_1,
    _variance_fn=VARIANCE_1
)


class TestClient(TestDataType):

    def test_inputs(self):
        pass

    def test_hashing(self):
        # test equal hash
        c_equal = InternalCustomer(name='c', commodity='c_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2)
        self.assertEqual(CUSTOMER, c_equal, msg="Nodes with the same name should be considered equal")
        p_equal = InternalPurchaser(name='p', commodity='p_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2)
        self.assertEqual(PURCHASER, p_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        c_diff = InternalCustomer(name='cd', commodity='c_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1)
        self.assertNotEqual(CUSTOMER, c_diff, msg="Nodes with different names should be considered different")
        p_diff = InternalPurchaser(name='pd', commodity='p_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1)
        self.assertNotEqual(CUSTOMER, p_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(CUSTOMER.key, 'c', msg="Wrong customer key name stored")
        self.assertEqual(PURCHASER.key, 'p', msg="Wrong purchaser key name stored")
        self.assertEqual(CUSTOMER.kind, 'client', msg="Customer node is not labelled as client")
        self.assertEqual(PURCHASER.kind, 'client', msg="Purchaser node is not labelled as client")
        self.assertFalse(CUSTOMER.purchaser, msg="Customer node is not labelled as non-purchaser")
        self.assertTrue(PURCHASER.purchaser, msg="Purchaser node is not labelled as purchaser")
        self.assertEqual(CUSTOMER.commodity_in, 'c_com', msg="Wrong customer inputs stored")
        self.assertEqual(PURCHASER.commodity_in, 'p_com', msg="Wrong purchaser inputs stored")
        self.assertSetEqual(CUSTOMER.commodities_out, set(), msg="Wrong customer outputs stored")
        self.assertSetEqual(PURCHASER.commodities_out, set(), msg="Wrong purchaser outputs stored")

    def test_exposed(self):
        c = CUSTOMER.exposed
        p = PURCHASER.exposed
        self.assertIsInstance(c, Customer, msg="Wrong exposed type")
        self.assertIsInstance(p, Purchaser, msg="Wrong exposed type")
        # test stored information
        self.assertEqual(c.name, 'c', msg="Wrong exposed name")
        self.assertEqual(p.name, 'p', msg="Wrong exposed name")
        self.assertEqual(c.commodity_in, 'c_com', msg="Wrong exposed inputs")
        self.assertEqual(p.commodity_in, 'p_com', msg="Wrong exposed inputs")
        self.assertSetEqual(c.commodities_out, set(), msg="Wrong exposed outputs")
        self.assertSetEqual(p.commodities_out, set(), msg="Wrong exposed outputs")
        self.assertDictEqual(c.demands.to_dict(), SERIES_1.to_dict(), msg='Wrong exposed demands')
        self.assertDictEqual(p.prices.to_dict(), SERIES_1.to_dict(), msg='Wrong exposed prices')
        # test dict (predictions need to be tested separately due to errors in the equality check)
        c_dict = c.dict
        c_pred = c_dict.pop('predictions')
        self.assertEqual(c_dict, {
            'name': 'c',
            'commodity_in': 'c_com',
            'commodities_out': set(),
        }, msg='Wrong dictionary returned for exposed customer')
        self.assertDictEqual(c_pred.to_dict(), SERIES_1.to_dict(), msg='Wrong dictionary returned for exposed customer')
        p_dict = p.dict
        p_pred = p_dict.pop('predictions')
        self.assertEqual(p_dict, {
            'name': 'p',
            'commodity_in': 'p_com',
            'commodities_out': set(),
        }, msg='Wrong dictionary returned for exposed purchaser')
        self.assertDictEqual(
            p_pred.to_dict(),
            SERIES_1.to_dict(),
            msg='Wrong dictionary returned for exposed purchaser'
        )
        # test immutability of mutable types
        c.demands[0] = 5.0
        self.assertEqual(c.demands[0], 5.0, msg="Exposed demands should be mutable")
        self.assertEqual(CUSTOMER.predictions[0], 3.0, msg="Internal demands should be immutable")
        p.prices[0] = 6.0
        self.assertEqual(p.prices[0], 6.0, msg="Exposed prices should be mutable")
        self.assertEqual(PURCHASER.predictions[0], 3.0, msg="Internal prices should be immutable")
