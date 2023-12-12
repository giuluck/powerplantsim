from ppsim.datatypes import InternalStorage, Storage
from test.datatypes.datatype import TestDataType, HORIZON

STORAGE = InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=1.0, _horizon=HORIZON)

CAPACITY_EXCEPTION = lambda v: f"Capacity should be strictly positive, got {v}"
DISSIPATION_EXCEPTION = lambda v: f"Dissipation should be in range [0, 1], got {v}"


class TestStorage(TestDataType):

    def test_inputs(self):
        # test correct dissipation
        InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=0.0, _horizon=HORIZON)
        InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=0.5, _horizon=HORIZON)
        InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=1.0, _horizon=HORIZON)
        # test incorrect dissipation
        with self.assertRaises(AssertionError, msg="Out of bound dissipation should raise exception") as e:
            InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=-1.0, _horizon=HORIZON)
        self.assertEqual(
            str(e.exception),
            DISSIPATION_EXCEPTION(-1.0),
            msg='Wrong exception message returned for out of bound dissipation on storage'
        )
        with self.assertRaises(AssertionError, msg="Out of bound dissipation should raise exception") as e:
            InternalStorage(name='s', commodity='s_com', capacity=100, dissipation=2.0, _horizon=HORIZON)
        self.assertEqual(
            str(e.exception),
            DISSIPATION_EXCEPTION(2.0),
            msg='Wrong exception message returned for out of bound dissipation on storage'
        )
        # test incorrect capacity
        with self.assertRaises(AssertionError, msg="Null capacity should raise exception") as e:
            InternalStorage(name='s', commodity='s_com', capacity=0.0, dissipation=1.0, _horizon=HORIZON)
        self.assertEqual(
            str(e.exception),
            CAPACITY_EXCEPTION(0.0),
            msg='Wrong exception message returned for null capacity on storage'
        )
        with self.assertRaises(AssertionError, msg="Negative capacity should raise exception") as e:
            InternalStorage(name='s', commodity='s_com', capacity=-10.0, dissipation=1.0, _horizon=HORIZON)
        self.assertEqual(
            str(e.exception),
            CAPACITY_EXCEPTION(-10.0),
            msg='Wrong exception message returned for negative capacity on storage'
        )

    def test_hashing(self):
        # test equal hash
        s_equal = InternalStorage(name='s', commodity='s_com_2', capacity=50, dissipation=1.0, _horizon=HORIZON)
        self.assertEqual(STORAGE, s_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        s_diff = InternalStorage(name='sd', commodity='s_com', capacity=100, dissipation=1.0, _horizon=HORIZON)
        self.assertNotEqual(STORAGE, s_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(STORAGE.key, 's', msg="Wrong storage key name stored")
        self.assertEqual(STORAGE.kind, 'storage', msg="Storage node is not labelled as supplier")
        self.assertEqual(STORAGE.commodity_in, 's_com', msg="Wrong storage inputs stored")
        self.assertSetEqual(STORAGE.commodities_out, {'s_com'}, msg="Wrong storage outputs stored")

    def test_exposed(self):
        s = STORAGE.exposed
        self.assertIsInstance(s, Storage, msg="Wrong exposed type")
        # test stored information
        self.assertEqual(s.name, 's', msg="Wrong exposed name")
        self.assertEqual(s.commodity_in, 's_com', msg="Wrong exposed inputs")
        self.assertSetEqual(s.commodities_out, {'s_com'}, msg="Wrong exposed outputs")
        self.assertEqual(s.capacity, 100.0, msg='Wrong exposed capacity')
        self.assertEqual(s.dissipation, 1.0, msg='Wrong exposed dissipation')
        # test dict
        self.assertEqual(s.dict, {
            'name': 's',
            'commodity_in': 's_com',
            'commodities_out': {'s_com'},
            'dissipation': 1.0,
            'capacity': 100.0
        }, msg='Wrong dictionary returned for exposed storage')
