import pyomo.environ as pyo

from powerplantsim.datatypes import Storage
from test.datatypes.test_datatype import TestDataType, PLANT, dummy_edge
from test.test_utils import SOLVER

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
INFINITE_CAPACITY_EXCEPTION = lambda: f"Capacity should be a finite number, got inf"
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
        with self.assertRaises(AssertionError, msg="Infinite capacity should raise exception") as e:
            Storage(
                name='s',
                commodity='s_com',
                capacity=float('inf'),
                dissipation=0.0,
                charge_rate=10.0,
                discharge_rate=12.0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            INFINITE_CAPACITY_EXCEPTION(),
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
        self.assertDictEqual(s_dict, {
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
        self.assertAlmostEqual(
            s.current_storage,
            0.0,
            msg=f"Storage current storage should be stored after update"
        )
        s.step(states={}, flows={
            dummy_edge(source='input_1', destination='s', commodity='s_com'): 1.0,
            dummy_edge(source='input_2', destination='s', commodity='s_com'): 2.0
        })
        self.assertDictEqual(s.storage.to_dict(), {0: 3.0}, msg=f"Storage storage should be filled after step")
        self.assertIsNone(s.current_storage, msg=f"Storage current storage should be None outside of the simulation")
        # test next step
        s.update(rng=None, flows={}, states={})
        self.assertAlmostEqual(
            s.current_storage,
            2.7,
            msg=f"Current storage should be computed based on previous storage"
        )
        # test input/output exception
        with self.assertRaises(AssertionError, msg="Input and output flows in storage should raise exception") as e:
            s.step(states={}, flows={
                dummy_edge(destination='s', commodity='s_com'): 1.0,
                dummy_edge(source='s', commodity='s_com'): 2.0
            })
        self.assertEqual(
            str(e.exception),
            INPUT_OUTPUT_EXCEPTION('s'),
            msg='Wrong exception message returned for input and output flows on storage'
        )
        # test charge rate exception
        with self.assertRaises(AssertionError, msg="Exceeding charge rate should raise exception") as e:
            s.step(states={}, flows={
                dummy_edge(source='input_1', destination='s', commodity='s_com'): 5.0,
                dummy_edge(source='input_2', destination='s', commodity='s_com'): 6.0
            })
        self.assertEqual(
            str(e.exception),
            CHARGE_RATE_EXCEPTION('s', 10.0, 11.0),
            msg='Wrong exception message returned for exceeding charge rate on storage'
        )
        # test discharge rate exception
        with self.assertRaises(AssertionError, msg="Exceeding discharge rate should raise exception") as e:
            s.step(states={}, flows={
                dummy_edge(source='s', destination='output_1', commodity='s_com'): 7.0,
                dummy_edge(source='s', destination='output_2', commodity='s_com'): 6.0
            })
        self.assertEqual(
            str(e.exception),
            DISCHARGE_RATE_EXCEPTION('s', 12.0, 13.0),
            msg='Wrong exception message returned for exceeding discharge rate on storage'
        )
        # test negative storage exception
        with self.assertRaises(AssertionError, msg="Negative storage should raise exception") as e:
            s.step(states={}, flows={dummy_edge(source='s', commodity='s_com'): 5.0})
        self.assertEqual(
            str(e.exception),
            NEGATIVE_STORAGE_EXCEPTION('s', -2.3),
            msg='Wrong exception message returned for negative storage on storage'
        )
        # test exceeded capacity exception
        with self.assertRaises(AssertionError, msg="Exceeded capacity should raise exception") as e:
            s.step(states={}, flows={dummy_edge(destination='s', commodity='s_com'): 3.0})
        self.assertEqual(
            str(e.exception),
            EXCEEDED_CAPACITY_EXCEPTION('s', 5.0, 5.7),
            msg='Wrong exception message returned for exceeded capacity on storage'
        )

    def test_pyomo(self):
        s = STORAGE.copy()
        s.update(rng=None, flows={}, states={})
        s.step(flows={dummy_edge(destination=s, commodity='s_com'): 2.0}, states={})
        s.update(rng=None, flows={}, states={})
        # test model
        model = s.to_pyomo(mutable=False)
        self.assertSetEqual(set(model.in_flows.keys()), {'s_com'}, msg="Wrong in_flows for storage block")
        self.assertSetEqual(set(model.out_flows.keys()), {'s_com'}, msg="Wrong out_flows for storage block")
        self.assertAlmostEqual(
            model.current_storage.value,
            1.8,
            msg="Wrong current storage initialized in storage block"
        )
        model.flows_value = pyo.Constraint(rule=model.in_flows['s_com'] == 3.0)
        results = pyo.SolverFactory(SOLVER).solve(model)
        self.assertEqual(
            results.solver.termination_condition,
            'optimal',
            msg="Storage block should be feasible if no constraint is passed"
        )
        self.assertAlmostEqual(
            pyo.value(model.storage),
            4.8,
            msg="Wrong storage computed for storage block"
        )
        with self.assertRaises(ValueError, msg="Accessing mutable storage parameters should raise exception"):
            pyo.value(s.to_pyomo(mutable=True).current_storage)
        # check capacity upper bound
        model = s.to_pyomo(mutable=False)
        model.flows_value = pyo.Constraint(rule=model.in_flows['s_com'] == 10.0)
        results = pyo.SolverFactory(SOLVER).solve(model)
        self.assertEqual(
            results.solver.termination_condition,
            'infeasible',
            msg="Input flow that exceeds capacity should result in an infeasible model"
        )
        # check capacity lower bound
        model = s.to_pyomo(mutable=False)
        model.flows_value = pyo.Constraint(rule=model.out_flows['s_com'] == 5.0)
        results = pyo.SolverFactory(SOLVER).solve(model)
        self.assertEqual(
            results.solver.termination_condition,
            'infeasible',
            msg="Output flow that exceeds storage should result in an infeasible model"
        )
        # check flows selector
        model = s.to_pyomo(mutable=False)
        model.in_flows_value = pyo.Constraint(rule=model.in_flows['s_com'] == 10.0)
        model.out_flows_value = pyo.Constraint(rule=model.out_flows['s_com'] == 5.0)
        results = pyo.SolverFactory(SOLVER).solve(model)
        self.assertEqual(
            results.solver.termination_condition,
            'infeasible',
            msg="Output flow that exceeds storage should result in an infeasible model"
        )
