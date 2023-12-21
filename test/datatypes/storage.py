from ppsim.datatypes import Storage
from test.datatypes.datatype import TestDataType, PLANT

STORAGE = Storage(
    name='s',
    commodity='s_com',
    capacity=5.0,
    dissipation=0.1,
    charge_rate=10.0,
    discharge_rate=12.0,
    _plant=PLANT
)

CAPACITY_EXCEPTION = lambda v: f"Capacity should be strictly positive, got {v}"
DISSIPATION_EXCEPTION = lambda v: f"Dissipation should be in range [0, 1], got {v}"
INPUT_OUTPUT_EXCEPTION = lambda n: f"Storage node '{n}' can have either input or output flows " \
                                   f"in a single time step, got both"
CHARGE_RATE_EXCEPTION = lambda n, r, f: f"Storage node '{n}' should have maximal input flow {r}, got {f}"
DISCHARGE_RATE_EXCEPTION = lambda n, r, f: f"Storage node '{n}' should have maximal output flow {r}, got {f}"
NEGATIVE_STORAGE_EXCEPTION = lambda n, s: f"Storage node '{n}' cannot contain negative amount, got {s}"
EXCEEDED_CAPACITY_EXCEPTION = lambda n, c, s: f"Storage node '{n}' cannot contain more than {c} amount, got {s}"


class TestStorage(TestDataType):

    def test_inputs(self):
        # test correct dissipation
        Storage(
            name='s',
            commodity='s_com',
            capacity=100,
            dissipation=0.0,
            charge_rate=10.0,
            discharge_rate=12.0,
            _plant=PLANT
        )
        Storage(
            name='s',
            commodity='s_com',
            capacity=100,
            dissipation=0.5,
            charge_rate=10.0,
            discharge_rate=12.0,
            _plant=PLANT
        )
        Storage(
            name='s',
            commodity='s_com',
            capacity=100,
            dissipation=1.0,
            charge_rate=10.0,
            discharge_rate=12.0,
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
                discharge_rate=12.0,
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
                discharge_rate=12.0,
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
                discharge_rate=12.0,
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
                discharge_rate=12.0,
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
            discharge_rate=12.0,
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
            'commodity': 's_com',
            'dissipation': 0.1,
            'capacity': 5.0,
            'charge_rate': 10.0,
            'discharge_rate': 12.0,
            'current_storage': None
        }, msg='Wrong dictionary returned for storage')
        self.assertDictEqual(s_storage.to_dict(), {}, msg='Wrong dictionary returned for storage')

    def test_operation(self):
        # test basics
        s = STORAGE.copy()
        self.assertDictEqual(s.storage.to_dict(), {}, msg=f"Storage storage should be empty before the simulation")
        self.assertIsNone(s.current_storage, msg=f"Storage current storage should be None outside of the simulation")
        s.update(rng=None, flows={}, states={})
        self.assertDictEqual(s.storage.to_dict(), {}, msg=f"Storage storage should be empty before step")
        self.assertEqual(s.current_storage, 0.0, msg=f"Storage current storage should be stored after update")
        s.step(flows={('input_1', 's', 's_com'): 1.0, ('input_2', 's', 's_com'): 2.0}, states={})
        self.assertDictEqual(s.storage.to_dict(), {0: 3.0}, msg=f"Storage storage should be filled after step")
        self.assertIsNone(s.current_storage, msg=f"Storage current storage should be None outside of the simulation")
        # test next step
        s.update(rng=None, flows={}, states={})
        self.assertEqual(s.current_storage, 2.7, msg=f"Current storage should be computed based on previous storage")
        # test input/output exception
        with self.assertRaises(AssertionError, msg="Input and output flows in storage should raise exception") as e:
            s.step(flows={('input', 's', 's_com'): 1.0, ('s', 'output', 's_com'): 2.0}, states={})
        self.assertEqual(
            str(e.exception),
            INPUT_OUTPUT_EXCEPTION('s'),
            msg='Wrong exception message returned for input and output flows on storage'
        )
        # test charge rate exception
        with self.assertRaises(AssertionError, msg="Exceeding charge rate should raise exception") as e:
            s.step(flows={('input_1', 's', 's_com'): 5.0, ('input_2', 's', 's_com'): 6.0}, states={})
        self.assertEqual(
            str(e.exception),
            CHARGE_RATE_EXCEPTION('s', 10.0, 11.0),
            msg='Wrong exception message returned for exceeding charge rate on storage'
        )
        # test discharge rate exception
        with self.assertRaises(AssertionError, msg="Exceeding discharge rate should raise exception") as e:
            s.step(flows={('s', 'output_1', 's_com'): 7.0, ('s', 'output_2', 's_com'): 6.0}, states={})
        self.assertEqual(
            str(e.exception),
            DISCHARGE_RATE_EXCEPTION('s', 12.0, 13.0),
            msg='Wrong exception message returned for exceeding discharge rate on storage'
        )
        # test negative storage exception
        with self.assertRaises(AssertionError, msg="Negative storage should raise exception") as e:
            s.step(flows={('s', 'output', 's_com'): 5.0}, states={})
        self.assertEqual(
            str(e.exception),
            NEGATIVE_STORAGE_EXCEPTION('s', -2.3),
            msg='Wrong exception message returned for negative storage on storage'
        )
        # test exceeded capacity exception
        with self.assertRaises(AssertionError, msg="Exceeded capacity should raise exception") as e:
            s.step(flows={('input', 's', 's_com'): 3.0}, states={})
        self.assertEqual(
            str(e.exception),
            EXCEEDED_CAPACITY_EXCEPTION('s', 5.0, 5.7),
            msg='Wrong exception message returned for exceeded capacity on storage'
        )
