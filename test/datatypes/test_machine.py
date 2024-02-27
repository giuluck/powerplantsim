import numpy as np
import pyomo.environ as pyo

from powerplantsim.datatypes import Machine
from test.datatypes.test_datatype import TestDataType, SETPOINT, PLANT, dummy_edge
from test.test_utils import SOLVER

DISCRETE_MACHINE = Machine(
    name='m',
    _setpoint=SETPOINT,
    discrete_setpoint=True,
    max_starting=None,
    cost=0,
    _plant=PLANT
)

CONTINUOUS_MACHINE = Machine(
    name='m',
    _setpoint=SETPOINT,
    discrete_setpoint=False,
    max_starting=(1, 3),
    cost=0,
    _plant=PLANT
)

COST_EXCEPTION = lambda v: f"The operating cost of the machine must be non-negative, got {v}"
MAX_STARTING_EXCEPTION = lambda n: f"The number of starting must be a strictly positive integer, got {n}"
TIMESTEP_STARTING_EXCEPTION = \
    lambda n, t: f"The number of starting must be strictly less than the number of time steps, got {n} >= {t}"
SETPOINT_INDEX_EXCEPTION = lambda v: f"Setpoints should be non-negative, got {v}"
SETPOINT_FLOWS_EXCEPTION = lambda c, v: f"Setpoint flows should be non-negative, got {c}: {v}"
NONZERO_OFF_FLOWS_EXCEPTION = \
    lambda k, v, c, n: f"Got non-zero {k} flow {v} for '{c}' despite null setpoint for machine '{n}'"
UNSUPPORTED_STATE_EXCEPTION = lambda s, m: f"Unsupported state {s} for machine '{m}'"
WRONG_FLOW_EXCEPTION = lambda e, m, s, f: f"Flow {e} expected for machine '{m}' with state {s}, got {f}"
TOO_MANY_STARTING_EXCEPTION = lambda m, n, t: f"Machine '{m}' cannot be started for more than {n} times in {t} steps"


class TestMachine(TestDataType):

    def test_inputs(self):
        # test incorrect cost
        with self.assertRaises(AssertionError, msg="Negative cost should raise exception") as e:
            Machine(
                name='m',
                _setpoint=SETPOINT,
                discrete_setpoint=True,
                max_starting=None,
                cost=-1.0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            COST_EXCEPTION(-1.0),
            msg='Wrong exception message returned for negative cost on machine'
        )
        # test incorrect max starting
        with self.assertRaises(AssertionError, msg="Negative max starting should raise exception") as e:
            Machine(
                name='m',
                _setpoint=SETPOINT,
                discrete_setpoint=True,
                max_starting=(-1, 3),
                cost=0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            MAX_STARTING_EXCEPTION(-1),
            msg='Wrong exception message returned for negative max starting on machine'
        )
        with self.assertRaises(AssertionError, msg="Null max starting should raise exception") as e:
            Machine(
                name='m',
                _setpoint=SETPOINT,
                discrete_setpoint=True,
                max_starting=(0, 3),
                cost=0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            MAX_STARTING_EXCEPTION(0),
            msg='Wrong exception message returned for null max starting on machine'
        )

        with self.assertRaises(AssertionError, msg="Max starting lower than time steps should raise exception") as e:
            Machine(
                name='m',
                _setpoint=SETPOINT,
                discrete_setpoint=True,
                max_starting=(5, 3),
                cost=0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            TIMESTEP_STARTING_EXCEPTION(5, 3),
            msg='Wrong exception message returned for max starting lower than time steps on machine'
        )
        # test incorrect index setpoint (null input flow should not raise exception anymore)
        sp2 = SETPOINT.copy()
        sp2.index = [50.0, 0.0, 100.0]
        Machine(
            name='m',
            _setpoint=sp2,
            discrete_setpoint=False,
            max_starting=None,
            cost=0,
            _plant=PLANT
        )
        sp2.index = [50.0, -1.0, 100.0]
        with self.assertRaises(AssertionError, msg="Negative input flows in setpoint should raise exception") as e:
            Machine(
                name='m',
                _setpoint=sp2,
                discrete_setpoint=False,
                max_starting=None,
                cost=0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            SETPOINT_INDEX_EXCEPTION(-1.0),
            msg='Wrong exception message returned for negative setpoint on machine'
        )
        # test incorrect output flows
        sp2 = SETPOINT.copy()
        sp2[('input', 'in_com')] = [50.0, -10.0, 100.0]
        with self.assertRaises(AssertionError, msg="Negative output flows in setpoint should raise exception") as e:
            Machine(
                name='m',
                _setpoint=sp2,
                discrete_setpoint=False,
                max_starting=None,
                cost=0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            SETPOINT_FLOWS_EXCEPTION(('input', 'in_com'), -10.0),
            msg='Wrong exception message returned for negative output flows on machine'
        )
        sp2 = SETPOINT.copy()
        sp2[('output', 'out_com_1')] = [-1.0, 0.0, 0.5]
        with self.assertRaises(AssertionError, msg="Negative output flows in setpoint should raise exception") as e:
            Machine(
                name='m',
                _setpoint=sp2,
                discrete_setpoint=False,
                max_starting=None,
                cost=0,
                _plant=PLANT
            )
        self.assertEqual(
            str(e.exception),
            SETPOINT_FLOWS_EXCEPTION(('output', 'out_com_1'), -1.0),
            msg='Wrong exception message returned for negative output flows on machine'
        )

    def test_hashing(self):
        # test equal hash
        m_equal = Machine(
            name='m',
            _setpoint=SETPOINT.copy(),
            discrete_setpoint=True,
            max_starting=(1, 7),
            cost=100,
            _plant=PLANT
        )
        self.assertEqual(DISCRETE_MACHINE, m_equal, msg="Nodes with the same name should be considered equal")
        # test different hash
        m_diff = Machine(
            name='md',
            _setpoint=SETPOINT,
            discrete_setpoint=False,
            max_starting=None,
            cost=0,
            _plant=PLANT
        )
        self.assertNotEqual(DISCRETE_MACHINE, m_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        self.assertEqual(DISCRETE_MACHINE.key, 'm', msg="Wrong machine key name stored")
        self.assertEqual(DISCRETE_MACHINE.kind, 'machine', msg="Machine node is not labelled as machine")
        self.assertSetEqual(DISCRETE_MACHINE.commodities_in, {'in_com'}, msg="Wrong machine inputs stored")
        self.assertSetEqual(
            DISCRETE_MACHINE.commodities_out,
            {'out_com_1', 'out_com_2'},
            msg="Wrong machine outputs stored"
        )
        self.assertDictEqual(DISCRETE_MACHINE.setpoint.to_dict(), SETPOINT.to_dict(), msg='Wrong setpoint stored')

    def test_immutability(self):
        DISCRETE_MACHINE.states[0] = 5.0
        DISCRETE_MACHINE.setpoint.iloc[0, 0] = 5.0
        self.assertEqual(len(DISCRETE_MACHINE.states), 0, msg="Machine states should be immutable")
        self.assertAlmostEqual(
            DISCRETE_MACHINE.setpoint.iloc[0, 0],
            50.0,
            msg="Machine setpoint should be immutable"
        )

    def test_dict(self):
        # pandas series and dataframes need to be tested separately due to errors in the equality check
        m_dict = DISCRETE_MACHINE.dict
        m_states = m_dict.pop('states')
        m_setpoint = m_dict.pop('setpoint')
        self.assertDictEqual(m_dict, {
            'name': 'm',
            'kind': 'machine',
            'commodities_in': {'in_com'},
            'commodities_out': {'out_com_1', 'out_com_2'},
            'discrete_setpoint': True,
            'max_starting': None,
            'cost': 0,
            'current_state': None
        }, msg='Wrong dictionary returned for machine')
        self.assertDictEqual(m_states.to_dict(), {}, msg='Wrong dictionary returned for machine')
        self.assertDictEqual(m_setpoint.to_dict(), SETPOINT.to_dict(), msg='Wrong dictionary returned for machine')

    def test_operation(self):
        # test basics
        m = DISCRETE_MACHINE.copy()
        self.assertDictEqual(m.states.to_dict(), {}, msg=f"Machine states should be empty before the simulation")
        self.assertIsNone(m.current_state, msg=f"Machine current state should be None outside of the simulation")
        m.update(rng=None, flows={}, states={m: 0.5})
        self.assertDictEqual(m.states.to_dict(), {}, msg=f"Machine states should be empty before step")
        self.assertAlmostEqual(
            m.current_state,
            0.5,
            msg=f"Machine current state should be stored after update"
        )
        m.step(states={m: 1.0}, flows={
            dummy_edge(destination=m, commodity='in_com'): 100.0,
            dummy_edge(source=m, commodity='out_com_1'): 1.0,
            dummy_edge(source=m, commodity='out_com_2'): 60.0
        })
        self.assertDictEqual(m.states.to_dict(), {0: 1.0}, msg=f"Machine states should be filled after step")
        self.assertIsNone(m.current_state, msg=f"Machine current state should be None outside of the simulation")
        # test null state
        m = DISCRETE_MACHINE.copy()
        m.step(states={m: np.nan}, flows={
            dummy_edge(destination=m, commodity='in_com'): 0.0,
            dummy_edge(source=m, commodity='out_com_1'): 0.0,
            dummy_edge(source=m, commodity='out_com_2'): 0.0
        })
        self.assertDictEqual(m.states.isna().to_dict(), {0: True}, msg=f"Wrong machine states stored after null state")
        m = CONTINUOUS_MACHINE.copy()
        m.step(states={m: np.nan}, flows={
            dummy_edge(destination=m, commodity='in_com'): 0.0,
            dummy_edge(source=m, commodity='out_com_1'): 0.0,
            dummy_edge(source=m, commodity='out_com_2'): 0.0
        })
        self.assertDictEqual(m.states.isna().to_dict(), {0: True}, msg=f"Wrong machine states stored after null state")
        m = DISCRETE_MACHINE.copy()
        with self.assertRaises(AssertionError, msg="Non-zero flows after null state should raise exception") as e:
            m.step(states={m: np.nan}, flows={
                dummy_edge(destination=m, commodity='in_com'): 10.0,
                dummy_edge(source=m, commodity='out_com_1'): 0.0,
                dummy_edge(source=m, commodity='out_com_2'): 0.0
            })
        self.assertEqual(
            str(e.exception),
            NONZERO_OFF_FLOWS_EXCEPTION('input', 10.0, 'in_com', 'm'),
            msg='Wrong exception message returned for on-zero flows after null state on machine'
        )
        m = CONTINUOUS_MACHINE.copy()
        with self.assertRaises(AssertionError, msg="Non-zero flows after null state should raise exception") as e:
            m.step(states={m: np.nan}, flows={
                dummy_edge(destination=m, commodity='in_com'): 0.0,
                dummy_edge(source=m, commodity='out_com_1'): 10.0,
                dummy_edge(source=m, commodity='out_com_2'): 0.0
            })
        self.assertEqual(
            str(e.exception),
            NONZERO_OFF_FLOWS_EXCEPTION('output', 10.0, 'out_com_1', 'm'),
            msg='Wrong exception message returned for on-zero flows after null state on machine'
        )
        # test discrete setpoint
        # noinspection DuplicatedCode
        m = DISCRETE_MACHINE.copy()
        m.step(states={m: 0.5}, flows={
            dummy_edge(destination=m, commodity='in_com'): 50.0,
            dummy_edge(source=m, commodity='out_com_1'): 0.0,
            dummy_edge(source=m, commodity='out_com_2'): 10.0
        })
        self.assertDictEqual(m.states.to_dict(), {0: 0.5}, msg=f"Wrong machine states stored with discrete setpoint")
        m = DISCRETE_MACHINE.copy()
        m.step(states={m: 0.75}, flows={
            dummy_edge(destination=m, commodity='in_com'): 75.0,
            dummy_edge(source=m, commodity='out_com_1'): 0.5,
            dummy_edge(source=m, commodity='out_com_2'): 30.0
        })
        self.assertDictEqual(m.states.to_dict(), {0: 0.75}, msg=f"Wrong machine states stored with discrete setpoint")
        m = DISCRETE_MACHINE.copy()
        m.step(states={m: 1.0}, flows={
            dummy_edge(destination=m, commodity='in_com'): 100.0,
            dummy_edge(source=m, commodity='out_com_1'): 1.0,
            dummy_edge(source=m, commodity='out_com_2'): 60.0
        })
        self.assertDictEqual(m.states.to_dict(), {0: 1.0}, msg=f"Wrong machine states stored with discrete setpoint")
        with self.assertRaises(AssertionError, msg="Unexpected discrete setpoint should raise exception") as e:
            m.step(states={m: 0.7}, flows={})
        self.assertEqual(
            str(e.exception),
            UNSUPPORTED_STATE_EXCEPTION(0.7, 'm'),
            msg='Wrong exception message returned for discrete setpoint operation on machine'
        )
        with self.assertRaises(AssertionError, msg="Unexpected discrete setpoint should raise exception") as e:
            m.step(states={m: 0.2}, flows={})
        self.assertEqual(
            str(e.exception),
            UNSUPPORTED_STATE_EXCEPTION(0.2, 'm'),
            msg='Wrong exception message returned for discrete setpoint operation on machine'
        )
        with self.assertRaises(AssertionError, msg="Unexpected discrete setpoint should raise exception") as e:
            m.step(states={m: 1.3}, flows={})
        self.assertEqual(
            str(e.exception),
            UNSUPPORTED_STATE_EXCEPTION(1.3, 'm'),
            msg='Wrong exception message returned for discrete setpoint operation on machine'
        )
        # test continuous setpoint
        # noinspection DuplicatedCode
        m = CONTINUOUS_MACHINE.copy()
        m.step(states={m: 0.5}, flows={
            dummy_edge(destination=m, commodity='in_com'): 50.0,
            dummy_edge(source=m, commodity='out_com_1'): 0.0,
            dummy_edge(source=m, commodity='out_com_2'): 10.0
        })
        self.assertDictEqual(m.states.to_dict(), {0: 0.5}, msg=f"Wrong machine states stored with continuous setpoint")
        m = CONTINUOUS_MACHINE.copy()
        m.step(states={m: 0.75}, flows={
            dummy_edge(destination=m, commodity='in_com'): 75.0,
            dummy_edge(source=m, commodity='out_com_1'): 0.5,
            dummy_edge(source=m, commodity='out_com_2'): 30.0
        })
        self.assertDictEqual(m.states.to_dict(), {0: 0.75}, msg=f"Wrong machine states stored with continuous setpoint")
        m = CONTINUOUS_MACHINE.copy()
        m.step(states={m: 1.0}, flows={
            dummy_edge(destination=m, commodity='in_com'): 100.0,
            dummy_edge(source=m, commodity='out_com_1'): 1.0,
            dummy_edge(source=m, commodity='out_com_2'): 60.0
        })
        self.assertDictEqual(m.states.to_dict(), {0: 1.0}, msg=f"Wrong machine states stored with continuous setpoint")
        m = CONTINUOUS_MACHINE.copy()
        m.step(states={m: 0.7}, flows={
            dummy_edge(destination=m, commodity='in_com'): 70.0,
            dummy_edge(source=m, commodity='out_com_1'): 0.4,
            dummy_edge(source=m, commodity='out_com_2'): 26.0
        })
        self.assertDictEqual(m.states.to_dict(), {0: 0.7}, msg=f"Wrong machine states stored with continuous setpoint")
        with self.assertRaises(AssertionError, msg="Unexpected continuous setpoint should raise exception") as e:
            m.step(states={m: 0.2}, flows={})
        self.assertEqual(
            str(e.exception),
            UNSUPPORTED_STATE_EXCEPTION(0.2, 'm'),
            msg='Wrong exception message returned for continuous setpoint operation on machine'
        )
        with self.assertRaises(AssertionError, msg="Unexpected continuous setpoint should raise exception") as e:
            m.step(states={m: 1.3}, flows={})
        self.assertEqual(
            str(e.exception),
            UNSUPPORTED_STATE_EXCEPTION(1.3, 'm'),
            msg='Wrong exception message returned for continuous setpoint operation on machine'
        )
        # test max starting
        m = CONTINUOUS_MACHINE.copy()
        m.step(states={m: 1.0}, flows={
            dummy_edge(destination=m, commodity='in_com'): 100.0,
            dummy_edge(source=m, commodity='out_com_1'): 1.0,
            dummy_edge(source=m, commodity='out_com_2'): 60.0
        })
        m.step(states={m: np.nan}, flows={})
        with self.assertRaises(AssertionError, msg="Too many starting should raise exception") as e:
            m.step(states={m: 0.5}, flows={
                dummy_edge(destination=m, commodity='in_com'): 50.0,
                dummy_edge(source=m, commodity='out_com_1'): 0.0,
                dummy_edge(source=m, commodity='out_com_2'): 10.0
            })
        self.assertEqual(
            str(e.exception),
            TOO_MANY_STARTING_EXCEPTION('m', 1, 3),
            msg='Wrong exception message returned for continuous setpoint operation on machine'
        )

    def test_pyomo(self):
        # test continuous/discrete machine & feasible/infeasible state values
        tests = {
            ('discrete', True): (DISCRETE_MACHINE, [None, 0.5, 0.75, 1.0]),
            ('discrete', False): (DISCRETE_MACHINE, [0.2, 0.6, 0.8, 1.2]),
            ('continuous', True): (CONTINUOUS_MACHINE, [None, 0.5, 0.6, 0.75, 0.8, 1.0]),
            ('continuous', False): (CONTINUOUS_MACHINE, [0.2, 1.2])
        }
        for (kind, feasible), (machine, values) in tests.items():
            # test model
            m = machine.copy()
            m.update(rng=None, flows={}, states={m: 0.75})
            model = m.to_pyomo(mutable=False)
            # in/out flows
            self.assertSetEqual(set(model.in_flows.keys()), {'in_com'}, msg=f"Wrong in_flows for {kind} machine block")
            self.assertSetEqual(
                set(model.out_flows.keys()),
                {'out_com_1', 'out_com_2'},
                msg=f"Wrong out_flows for {kind} machine block"
            )
            # state
            self.assertIsInstance(model.state, pyo.Var, msg=f"Wrong type variable for {kind} machine state")
            self.assertEqual(
                model.state.domain,
                pyo.NonNegativeReals,
                msg=f"Wrong variable domain stored for {kind} machine state."
            )
            self.assertEqual(
                model.state.value,
                0.75,
                msg=f"{kind.title()} machine state should be initialized to current state"
            )
            self.assertIsNone(
                m.to_pyomo(mutable=True).state.value,
                msg=f"{kind.title()} machine switch should not be initialized"
            )
            # switch
            self.assertIsInstance(model.switch, pyo.Var, msg=f"Wrong type variable for {kind} machine switch")
            self.assertEqual(
                model.switch.domain,
                pyo.Binary,
                msg=f"Wrong variable domain stored for {kind} machine switch."
            )
            self.assertEqual(
                model.switch.value,
                0,
                msg=f"{kind.title()} machine switch should be initialized to off"
            )
            self.assertEqual(
                m.to_pyomo(mutable=True).switch.value,
                0,
                msg=f"{kind.title()} machine switch should be initialized to off when mutable=True as well"
            )
            # test state constraint
            for value in values:
                m = machine.copy()
                m.update(rng=None, flows={}, states={m: 0.75})
                model = m.to_pyomo(mutable=False)
                if value is None:
                    switch, state = 0, None
                    model.switch_value = pyo.Constraint(rule=model.switch == 0)
                else:
                    switch, state = 1, value
                    model.switch_value = pyo.Constraint(rule=model.switch == 1)
                    model.state_value = pyo.Constraint(rule=model.state == value)
                results = pyo.SolverFactory(SOLVER).solve(model)
                if feasible:
                    self.assertEqual(
                        results.solver.termination_condition,
                        'optimal',
                        msg=f"{kind.title()} machine should be feasible when value is {value}"
                    )
                    self.assertEqual(
                        model.switch.value,
                        switch,
                        msg=f"Wrong switch computed for {kind} machine when value is {value}"
                    )
                    if state is not None:
                        self.assertEqual(
                            model.state.value,
                            state,
                            msg=f"Wrong state computed for {kind} machine when value is {value}"
                        )
                        state = np.nan if np.isclose(model.switch.value, 0.0) else model.state.value
                        for key, commodity in [('input', 'in_com'), ('output', 'out_com_1'), ('output', 'out_com_2')]:
                            expected = np.interp(state, xp=m.setpoint.index, fp=m.setpoint[(key, commodity)])
                            variables = model.in_flows if key == 'input' else model.out_flows
                            self.assertAlmostEqual(
                                variables[commodity].value,
                                expected,
                                msg=f"Wrong {key} flows computed for commodity {commodity} in {kind} machine block"
                            )
                    else:
                        for key, commodity in [('input', 'in_com'), ('output', 'out_com_1'), ('output', 'out_com_2')]:
                            variables = model.in_flows if key == 'input' else model.out_flows
                            self.assertAlmostEqual(
                                variables[commodity].value,
                                0.0,
                                msg=f"Wrong {key} flows computed for commodity {commodity} in {kind} machine block"
                            )
                else:
                    self.assertEqual(
                        results.solver.termination_condition,
                        'infeasible',
                        msg=f"{kind.title()} machine should be infeasible when value is {value}"
                    )
        # test max starting
        m = CONTINUOUS_MACHINE.copy()
        m.step(states={m: 1.0}, flows={
            dummy_edge(destination=m, commodity='in_com'): 100.0,
            dummy_edge(source=m, commodity='out_com_1'): 1.0,
            dummy_edge(source=m, commodity='out_com_2'): 60.0
        })
        m.step(states={m: np.nan}, flows={})
        m.update(rng=None, flows={}, states={m: 0.75})
        model = m.to_pyomo(mutable=False)
        model.state_value = pyo.Constraint(rule=model.switch == 1)
        results = pyo.SolverFactory(SOLVER).solve(model)
        self.assertEqual(
            results.solver.termination_condition,
            'infeasible',
            msg=f"Machine should be infeasible when max starting is exceeded and switch is set to on"
        )
