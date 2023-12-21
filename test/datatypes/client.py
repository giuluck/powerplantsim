import numpy as np

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

EXCEEDING_DEMAND_EXCEPTION = lambda n, d, f: f"Customer node '{n}' can accept at most {d} units, got {f}"

class TestClient(TestDataType):

    def test_inputs(self):
        pass

    def test_hashing(self):
        # test customer
        c_equal = Customer(name='c', commodity='c_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2, _plant=PLANT)
        self.assertEqual(CUSTOMER, c_equal, msg="Nodes with the same name should be considered equal")
        c_diff = Customer(name='cd', commodity='c_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1, _plant=PLANT)
        self.assertNotEqual(CUSTOMER, c_diff, msg="Nodes with different names should be considered different")
        # test purchaser
        p_equal = Purchaser(name='p', commodity='p_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2, _plant=PLANT)
        self.assertEqual(PURCHASER, p_equal, msg="Nodes with the same name should be considered equal")
        p_diff = Purchaser(name='pd', commodity='p_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1, _plant=PLANT)
        self.assertNotEqual(CUSTOMER, p_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        # test customer
        self.assertEqual(CUSTOMER.key, 'c', msg="Wrong customer key name stored")
        self.assertEqual(CUSTOMER.kind, 'client', msg="Customer node is not labelled as client")
        self.assertFalse(CUSTOMER.purchaser, msg="Customer node is not labelled as non-purchaser")
        self.assertSetEqual(CUSTOMER.commodities_in, {'c_com'}, msg="Wrong customer inputs stored")
        self.assertSetEqual(CUSTOMER.commodities_out, set(), msg="Wrong customer outputs stored")
        # test purchaser
        self.assertEqual(PURCHASER.key, 'p', msg="Wrong purchaser key name stored")
        self.assertEqual(PURCHASER.kind, 'client', msg="Purchaser node is not labelled as client")
        self.assertTrue(PURCHASER.purchaser, msg="Purchaser node is not labelled as purchaser")
        self.assertSetEqual(PURCHASER.commodities_in, {'p_com'}, msg="Wrong purchaser inputs stored")
        self.assertSetEqual(PURCHASER.commodities_out, set(), msg="Wrong purchaser outputs stored")

    def test_immutability(self):
        # test customer
        CUSTOMER.demands[0] = 5.0
        self.assertEqual(len(CUSTOMER.demands), 0, msg="Customer demands should be immutable")
        # test purchaser
        PURCHASER.prices[0] = 5.0
        self.assertEqual(len(PURCHASER.prices), 0, msg="Purchaser prices should be immutable")

    def test_dict(self):
        # pandas series need to be tested separately due to errors in the equality check
        # test customer
        c_dict = CUSTOMER.dict
        c_val = c_dict.pop('demands')
        self.assertEqual(c_dict, {
            'name': 'c',
            'kind': 'client',
            'purchaser': False,
            'commodity': 'c_com',
            'current_demand': None
        }, msg='Wrong dictionary returned for customer')
        self.assertDictEqual(c_val.to_dict(), {}, msg='Wrong dictionary returned for customer')
        # test purchaser
        p_dict = PURCHASER.dict
        p_val = p_dict.pop('prices')
        self.assertEqual(p_dict, {
            'name': 'p',
            'kind': 'client',
            'purchaser': True,
            'commodity': 'p_com',
            'current_price': None
        }, msg='Wrong dictionary returned for purchaser')
        self.assertDictEqual(p_val.to_dict(), {}, msg='Wrong dictionary returned for customer')

    def test_operation(self):
        # test customer
        c = CUSTOMER.copy()
        rng = np.random.default_rng(0)
        val = 3.0 + np.random.default_rng(0).normal()
        self.assertDictEqual(c.demands.to_dict(), {}, msg=f"Customer demands should be empty before the simulation")
        self.assertIsNone(c.current_demand, msg=f"Customer current demand should be None outside of the simulation")
        c.update(rng=rng, flows={}, states={})
        self.assertDictEqual(c.demands.to_dict(), {}, msg=f"Customer demands should be empty before step")
        self.assertEqual(c.current_demand, val, msg=f"Customer current demand should be stored after update")
        c.step(flows={}, states={})
        self.assertDictEqual(c.demands.to_dict(), {0: val}, msg=f"Customer demands should be filled after step")
        self.assertIsNone(c.current_demand, msg=f"Customer current demand should be None outside of the simulation")
        # test customer exception
        c = CUSTOMER.copy()
        rng = np.random.default_rng(0)
        c.update(rng=rng, flows={}, states={})
        with self.assertRaises(AssertionError, msg="Exceeding demand should raise exception") as e:
            c.step(states={}, flows={('input_1', 'c', 'c_com'): 1.0, ('input_2', 'c', 'c_com'): 3.0})
        self.assertEqual(
            str(e.exception),
            EXCEEDING_DEMAND_EXCEPTION('c', val, 4.0),
            msg='Wrong exception message returned for exceeding demand on customer'
        )
        # test purchaser
        p = PURCHASER.copy()
        rng = np.random.default_rng(0)
        val = 3.0 + np.random.default_rng(0).normal()
        self.assertDictEqual(p.prices.to_dict(), {}, msg=f"Purchaser prices should be empty before the simulation")
        self.assertIsNone(p.current_price, msg=f"Purchaser current price should be None outside of the simulation")
        p.update(rng=rng, flows={}, states={})
        self.assertDictEqual(p.prices.to_dict(), {}, msg=f"Purchaser prices should be empty before step")
        self.assertEqual(p.current_price, val, msg=f"Purchaser current price should be stored after update")
        p.step(flows={}, states={})
        self.assertDictEqual(p.prices.to_dict(), {0: val}, msg=f"Purchaser prices should be filled after step")
        self.assertIsNone(p.current_price, msg=f"Purchaser current price should be None outside of the simulation")

