from ppsim.datatypes import Customer, Purchaser
from test.datatypes.datatype import TestDataType, SERIES_1, SERIES_2, VARIANCE_1, VARIANCE_2, PLANT

CUSTOMER = Customer(
    name='c',
    commodity='c_com',
    _predictions=SERIES_1,
    _variance_fn=VARIANCE_1,
    _plant=PLANT
)

PURCHASER = Purchaser(
    name='p',
    commodity='p_com',
    _predictions=SERIES_1,
    _variance_fn=VARIANCE_1,
    _plant=PLANT
)


class TestClient(TestDataType):

    def test_inputs(self):
        pass

    def test_hashing(self):
        # test equal hash
        c_equal = Customer(name='c', commodity='c_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2, _plant=PLANT)
        self.assertEqual(CUSTOMER, c_equal, msg="Nodes with the same name should be considered equal")
        p_equal = Purchaser(name='p', commodity='p_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2, _plant=PLANT)
        self.assertEqual(PURCHASER, p_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        c_diff = Customer(name='cd', commodity='c_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1, _plant=PLANT)
        self.assertNotEqual(CUSTOMER, c_diff, msg="Nodes with different names should be considered different")
        p_diff = Purchaser(name='pd', commodity='p_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1, _plant=PLANT)
        self.assertNotEqual(CUSTOMER, p_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(CUSTOMER.key, 'c', msg="Wrong customer key name stored")
        self.assertEqual(PURCHASER.key, 'p', msg="Wrong purchaser key name stored")
        self.assertEqual(CUSTOMER.kind, 'client', msg="Customer node is not labelled as client")
        self.assertEqual(PURCHASER.kind, 'client', msg="Purchaser node is not labelled as client")
        self.assertFalse(CUSTOMER.purchaser, msg="Customer node is not labelled as non-purchaser")
        self.assertTrue(PURCHASER.purchaser, msg="Purchaser node is not labelled as purchaser")
        self.assertSetEqual(CUSTOMER.commodities_in, {'c_com'}, msg="Wrong customer inputs stored")
        self.assertSetEqual(PURCHASER.commodities_in, {'p_com'}, msg="Wrong purchaser inputs stored")
        self.assertSetEqual(CUSTOMER.commodities_out, set(), msg="Wrong customer outputs stored")
        self.assertSetEqual(PURCHASER.commodities_out, set(), msg="Wrong purchaser outputs stored")

    def test_immutability(self):
        CUSTOMER.demands[0] = 5.0
        CUSTOMER.predicted_demands[0] = 5.0
        self.assertEqual(len(CUSTOMER.demands), 0, msg="Customer demands should be immutable")
        self.assertEqual(CUSTOMER.predicted_demands[0], 3.0, msg="Customer predicted demands should be immutable")
        PURCHASER.prices[0] = 5.0
        PURCHASER.predicted_prices[0] = 5.0
        self.assertEqual(len(PURCHASER.prices), 0, msg="Purchaser prices should be immutable")
        self.assertEqual(PURCHASER.predicted_prices[0], 3.0, msg="Purchaser predicted prices should be immutable")

    def test_dict(self):
        # pandas series need to be tested separately due to errors in the equality check
        c_dict = CUSTOMER.dict
        c_val = c_dict.pop('demands')
        c_pred = c_dict.pop('predicted_demands')
        self.assertEqual(c_dict, {
            'name': 'c',
            'kind': 'client',
            'purchaser': False,
            'commodities_in': {'c_com'},
            'commodities_out': set(),
            'current_demand': None
        }, msg='Wrong dictionary returned for customer')
        self.assertDictEqual(c_val.to_dict(), {}, msg='Wrong dictionary returned for customer')
        self.assertDictEqual(c_pred.to_dict(), SERIES_1.to_dict(), msg='Wrong dictionary returned for customer')
        p_dict = PURCHASER.dict
        p_val = p_dict.pop('prices')
        p_pred = p_dict.pop('predicted_prices')
        self.assertEqual(p_dict, {
            'name': 'p',
            'kind': 'client',
            'purchaser': True,
            'commodities_in': {'p_com'},
            'commodities_out': set(),
            'current_price': None
        }, msg='Wrong dictionary returned for purchaser')
        self.assertDictEqual(p_val.to_dict(), {}, msg='Wrong dictionary returned for customer')
        self.assertDictEqual(p_pred.to_dict(), SERIES_1.to_dict(), msg='Wrong dictionary returned for purchaser')
