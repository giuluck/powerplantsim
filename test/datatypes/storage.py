from ppsim.datatypes import Storage
from test.datatypes.datatype import TestDataType, PLANT

STORAGE = Storage(
    name='s',
    commodity='s_com',
    capacity=100.0,
    dissipation=1.0,
    charge_rate=10.0,
    discharge_rate=10.0,
    _plant=PLANT
)

CAPACITY_EXCEPTION = lambda v: f"Capacity should be strictly positive, got {v}"
DISSIPATION_EXCEPTION = lambda v: f"Dissipation should be in range [0, 1], got {v}"


class TestStorage(TestDataType):

    def test_inputs(self):
        # test correct dissipation
        Storage(
            name='s',
            commodity='s_com',
            capacity=100,
            dissipation=0.0,
            charge_rate=10.0,
            discharge_rate=10.0,
            _plant=PLANT
        )
        Storage(
            name='s',
            commodity='s_com',
            capacity=100,
            dissipation=0.5,
            charge_rate=10.0,
            discharge_rate=10.0,
            _plant=PLANT
        )
        Storage(
            name='s',
            commodity='s_com',
            capacity=100,
            dissipation=1.0,
            charge_rate=10.0,
            discharge_rate=10.0,
            _plant=PLANT
        )
        # test incorrect dissipation
        with self.assertRaises(AssertionError, msg="Out of bound dissipation should raise exception") as e:
            Storage(
                name='s',
                commodity='s_com',
                capacity=100,
                dissipation=-1.0,
                charge_rate=10.0,
                discharge_rate=10.0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            DISSIPATION_EXCEPTION(-1.0),
            msg='Wrong exception message returned for out of bound dissipation on storage'
        )
        with self.assertRaises(AssertionError, msg="Out of bound dissipation should raise exception") as e:
            Storage(
                name='s',
                commodity='s_com',
                capacity=100,
                dissipation=2.0,
                charge_rate=10.0,
                discharge_rate=10.0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            DISSIPATION_EXCEPTION(2.0),
            msg='Wrong exception message returned for out of bound dissipation on storage'
        )
        # test incorrect capacity
        with self.assertRaises(AssertionError, msg="Null capacity should raise exception") as e:
            Storage(
                name='s',
                commodity='s_com',
                capacity=0.0,
                dissipation=0.0,
                charge_rate=10.0,
                discharge_rate=10.0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            CAPACITY_EXCEPTION(0.0),
            msg='Wrong exception message returned for null capacity on storage'
        )
        with self.assertRaises(AssertionError, msg="Negative capacity should raise exception") as e:
            Storage(
                name='s',
                commodity='s_com',
                capacity=-10.0,
                dissipation=0.0,
                charge_rate=10.0,
                discharge_rate=10.0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            CAPACITY_EXCEPTION(-10.0),
            msg='Wrong exception message returned for negative capacity on storage'
        )

    def test_hashing(self):
        # test equal hash
        s_equal = Storage(
            name='s',
            commodity='s_com_2',
            capacity=50.0,
            dissipation=1.0,
            charge_rate=5.0,
            discharge_rate=5.0,
            _plant=PLANT
        )
        self.assertEqual(STORAGE, s_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        s_diff = Storage(
            name='sd',
            commodity='s_com',
            capacity=100,
            dissipation=0.0,
            charge_rate=10.0,
            discharge_rate=10.0,
            _plant=PLANT
        )
        self.assertNotEqual(STORAGE, s_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(STORAGE.key, 's', msg="Wrong storage key name stored")
        self.assertEqual(STORAGE.kind, 'storage', msg="Storage node is not labelled as supplier")
        self.assertSetEqual(STORAGE.commodities_in, {'s_com'}, msg="Wrong storage inputs stored")
        self.assertSetEqual(STORAGE.commodities_out, {'s_com'}, msg="Wrong storage outputs stored")

    def test_immutability(self):
        STORAGE.storage[0] = 5.0
        self.assertEqual(len(STORAGE.storage), 0, msg="Storage storage should be immutable")

    def test_dict(self):
        s_dict = STORAGE.dict
        s_storage = s_dict.pop('storage')
        self.assertEqual(s_dict, {
            'name': 's',
            'kind': 'storage',
            'commodities_in': {'s_com'},
            'commodities_out': {'s_com'},
            'dissipation': 1.0,
            'capacity': 100.0,
            'charge_rate': 10.0,
            'discharge_rate': 10.0,
            'current_storage': None
        }, msg='Wrong dictionary returned for storage')
        self.assertDictEqual(s_storage.to_dict(), {}, msg='Wrong dictionary returned for storage')
