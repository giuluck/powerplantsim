import unittest

import numpy as np
import pandas as pd

from ppsim import Plant
from ppsim.plant.actions.action import RecourseAction
from ppsim.utils.typing import Plan

SETPOINT = {'setpoint': [1., 3.], 'input': {'in': [1., 3.]}, 'output': {'out': [1., 3.]}}

PLANT = Plant(horizon=3)
PLANT.add_supplier(name='sup', commodity='in', predictions=[1., 2., 3.])
PLANT.add_machine(name='mac_1', parents='sup', setpoint=SETPOINT)
PLANT.add_machine(name='mac_2', parents='sup', setpoint=SETPOINT)
PLANT.add_storage(name='sto', parents='mac_1', commodity='out')
PLANT.add_client(name='cus', parents=['sto', 'mac_1'], commodity='out', predictions=[1., 2., 3.])
PLANT.add_client(name='pur', parents=['sto', 'mac_2'], commodity='out', predictions=[1., 2., 3.], purchaser=True)

PLAN = {
    'mac_1': [1., 2., 3.],
    'mac_2': 1.,
    ('sup', 'mac_1'): [1., 2., 3.],
    ('sup', 'mac_2'): 1.,
    ('mac_1', 'cus'): [1., 2., 3.],
    ('mac_1', 'sto'): [1., 2., 3.],
    ('mac_2', 'pur'): [1., 2., 3.],
    ('sto', 'cus'): [1., 2., 3.],
    ('sto', 'pur'): [1., 2., 3.]
}

IMPLEMENTATION = {
    'mac_1': [1., 2., 3.],
    'mac_2': [np.nan, np.nan, np.nan],
    ('sup', 'mac_1'): [1., 2., 3.],
    ('sup', 'mac_2'): [0., 0., 0.],
    ('mac_1', 'cus'): [1., 2., 3.],
    ('mac_1', 'sto'): [0., 0., 0.],
    ('mac_2', 'pur'): [0., 0., 0.],
    ('sto', 'cus'): [0., 0., 0.],
    ('sto', 'pur'): [0., 0., 0.]
}

SECOND_RUN_EXCEPTION = lambda: "Simulation for this plant was already run, create a new instance to run another one"
INPUT_VECTOR_EXCEPTION = lambda k, l: f"Vector for key '{k}' has length {l}, expected 3"
UNKNOWN_DATATYPE_EXCEPTION = lambda k: f"Key {k} is not present in the plant"
MISSING_DATATYPE_EXCEPTION = lambda t, g, l: f"No {t} vector has been passed for {g} {l}"


class DummyAction(RecourseAction):
    def execute(self) -> Plan:
        return {key: vector[self._plant.step] for key, vector in IMPLEMENTATION.items()}


class TestPlantRun(unittest.TestCase):
    def test_output(self):
        """Test the output of a simulation."""
        p = PLANT.copy()
        output = p.run(plan=PLAN, action=DummyAction())
        self.assertDictEqual(output.flows.to_dict(), {
            ('sup', 'mac_1', 'in'): {0: 1., 1: 2., 2: 3.},
            ('sup', 'mac_2', 'in'): {0: 0., 1: 0., 2: 0.},
            ('mac_1', 'cus', 'out'): {0: 1., 1: 2., 2: 3.},
            ('mac_1', 'sto', 'out'): {0: 0., 1: 0., 2: 0.},
            ('mac_2', 'pur', 'out'): {0: 0., 1: 0., 2: 0.},
            ('sto', 'cus', 'out'): {0: 0., 1: 0., 2: 0.},
            ('sto', 'pur', 'out'): {0: 0., 1: 0., 2: 0.}
        }, msg=f"Wrong output flows returned")
        self.assertDictEqual(
            output.storage.to_dict(),
            {'sto': {0: 0., 1: 0., 2: 0.}},
            msg="Wrong output storage returned"
        )
        self.assertDictEqual(
            output.demands.to_dict(),
            {'cus': {0: 1., 1: 2., 2: 3.}},
            msg="Wrong output demands returned"
        )
        self.assertDictEqual(
            output.buying_prices.to_dict(),
            {'pur': {0: 1., 1: 2., 2: 3.}},
            msg="Wrong output buying prices returned"
        )
        self.assertDictEqual(
            output.sell_prices.to_dict(),
            {'sup': {0: 1., 1: 2., 2: 3.}},
            msg="Wrong output selling prices returned"
        )
        # test mac_1 and mac_2 separately due to nan values returning False when compared
        self.assertSetEqual(set(output.states.columns), {'mac_1', 'mac_2'}, msg=f"Wrong output states returned")
        self.assertDictEqual(
            output.states.to_dict()['mac_1'],
            {0: 1., 1: 2., 2: 3.},
            msg=f"Wrong output states returned"
        )
        self.assertDictEqual(
            output.states.isna().to_dict()['mac_2'],
            {0: True, 1: True, 2: True},
            msg=f"Wrong output states returned"
        )

    def test_already_run(self):
        p = PLANT.copy()
        p.run(plan=PLAN, action=DummyAction())
        with self.assertRaises(AssertionError, msg="Running a second simulation should raise an error") as e:
            p.run(plan=PLAN, action=DummyAction())
        self.assertEqual(
            str(e.exception),
            SECOND_RUN_EXCEPTION(),
            msg='Wrong exception message returned for running a second simulation on the same plant'
        )

    def test_input_plan(self):
        """Test that the input plan is consistency processed."""
        # test correct dataframe input
        p = PLANT.copy()
        df = pd.DataFrame(PLAN, index=p.horizon)
        p.run(plan=df, action=DummyAction())
        # test wrong input vector
        p = PLANT.copy()
        with self.assertRaises(AssertionError, msg="Wrong input vectors should raise an error") as e:
            p.run(plan={'mac_1': [1., 2.]}, action=DummyAction())
        self.assertEqual(
            str(e.exception),
            INPUT_VECTOR_EXCEPTION('mac_1', 2),
            msg='Wrong exception message returned for wrong input vectors on plant'
        )
        with self.assertRaises(AssertionError, msg="Wrong input vectors should raise an error") as e:
            p.run(plan={'mac_1': [1., 2., 3.], ('sup', 'mac_1'): [1.]}, action=DummyAction())
        self.assertEqual(
            str(e.exception),
            INPUT_VECTOR_EXCEPTION(('sup', 'mac_1'), 1),
            msg='Wrong exception message returned for wrong input vectors on plant'
        )
        # test unknown datatype
        with self.assertRaises(AssertionError, msg="Unknown datatype should raise an error") as e:
            p.run(plan={'mac': [1., 2., 3.]}, action=DummyAction())
        self.assertEqual(
            str(e.exception),
            UNKNOWN_DATATYPE_EXCEPTION("'mac'"),
            msg='Wrong exception message returned for unknown datatype vectors on plant'
        )
        with self.assertRaises(AssertionError, msg="Unknown datatype should raise an error") as e:
            p.run(plan={('sup', 'mac'): [1., 2., 3.]}, action=DummyAction())
        self.assertEqual(
            str(e.exception),
            UNKNOWN_DATATYPE_EXCEPTION(('sup', 'mac')),
            msg='Wrong exception message returned for unknown datatype vectors on plant'
        )
        # test missing datatype
        plan = PLAN.copy()
        plan.pop('mac_2')
        with self.assertRaises(AssertionError, msg="Missing datatype should raise an error") as e:
            p.run(plan=plan, action=DummyAction())
        self.assertEqual(
            str(e.exception),
            MISSING_DATATYPE_EXCEPTION('states', 'machines', ['mac_2']),
            msg='Wrong exception message returned for missing datatype vectors on plant'
        )
        plan = PLAN.copy()
        plan.pop(('sup', 'mac_2'))
        with self.assertRaises(AssertionError, msg="Missing datatype should raise an error") as e:
            p.run(plan=plan, action=DummyAction())
        self.assertEqual(
            str(e.exception),
            MISSING_DATATYPE_EXCEPTION('flows', 'edges', [('sup', 'mac_2')]),
            msg='Wrong exception message returned for missing datatype vectors on plant'
        )

    def test_action_output(self):
        """Test that the recourse action output is consistency processed."""
        p = PLANT.copy()
        # test unknown datatype
        with self.assertRaises(AssertionError, msg="Unknown datatype in recourse action should raise an error") as e:
            p.run(plan=PLAN, action=lambda _: {'mac': 1.})
        self.assertEqual(
            str(e.exception),
            UNKNOWN_DATATYPE_EXCEPTION("'mac'"),
            msg='Wrong exception message returned for unknown datatype vectors in recourse action on plant'
        )
        p = PLANT.copy()
        with self.assertRaises(AssertionError, msg="Unknown datatype in recourse action should raise an error") as e:
            p.run(plan=PLAN, action=lambda _: {('sup', 'mac'): 1.})
        self.assertEqual(
            str(e.exception),
            UNKNOWN_DATATYPE_EXCEPTION(('sup', 'mac')),
            msg='Wrong exception message returned for unknown datatype in recourse action vectors on plant'
        )
        # test missing datatype
        p = PLANT.copy()
        a = DummyAction().build(p)
        with self.assertRaises(AssertionError, msg="Missing datatype in recourse action should raise an error") as e:
            p.run(plan=PLAN, action=lambda _: {k: v for k, v in a.execute().items() if k != 'mac_2'})
        self.assertEqual(
            str(e.exception),
            MISSING_DATATYPE_EXCEPTION('states', 'machines', ['mac_2']),
            msg='Wrong exception message returned for missing datatype in recourse action vectors on plant'
        )
        p = PLANT.copy()
        a = DummyAction().build(p)
        with self.assertRaises(AssertionError, msg="Missing datatype in recourse action should raise an error") as e:
            p.run(
                plan=PLAN,
                action=lambda _: {k: v for k, v in a.execute().items() if k != ('sup', 'mac_2')}
            )
        self.assertEqual(
            str(e.exception),
            MISSING_DATATYPE_EXCEPTION('flows', 'edges', [('sup', 'mac_2')]),
            msg='Wrong exception message returned for missing datatype in recourse action vectors on plant'
        )
