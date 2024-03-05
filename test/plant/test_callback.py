import unittest
from typing import Dict, List, Union

import pandas as pd
from pyomo.common.errors import ApplicationError

from powerplantsim.datatypes import Node, Edge
from powerplantsim.datatypes.datatype import DataType
from powerplantsim.plant import Callback
from powerplantsim.utils.typing import State, Flow
from test.test_utils import PLANT, PLAN, SOLVER


class DummyCallback(Callback):

    def __init__(self):
        self.start: Dict[DataType, pd.Series] = {}
        self.updated: List[Dict[DataType, Union[State, Flow]]] = []
        self.calls: List[str] = []

    def on_simulation_start(self, plant, states: Dict[Node, pd.Series], flows: Dict[Edge, pd.Series]):
        self.start = {**states, **flows}
        self.calls.append('on_simulation_start')

    def on_iteration_update(self, plant):
        self.calls.append('on_iteration_update')

    def on_iteration_recourse(self, plant, states: Dict[Node, State], flows: Dict[Edge, Flow]):
        self.updated.append({**states, **flows})
        self.calls.append('on_iteration_recourse')

    def on_iteration_step(self, plant):
        self.calls.append('on_iteration_step')

    def on_simulation_end(self, plant):
        self.calls.append('on_simulation_end')


class TestCallback(unittest.TestCase):
    def test_callback(self):
        p = PLANT.copy()
        c = DummyCallback()
        try:
            p.run(plan=PLAN, action=SOLVER, callbacks=[c], progress=False)
            # check correct order of calls
            self.assertListEqual(
                c.calls,
                [
                    'on_simulation_start',
                    'on_iteration_update',
                    'on_iteration_recourse',
                    'on_iteration_step',
                    'on_iteration_update',
                    'on_iteration_recourse',
                    'on_iteration_step',
                    'on_iteration_update',
                    'on_iteration_recourse',
                    'on_iteration_step',
                    'on_simulation_end'
                ],
                msg="Callback hooks were not called correctly."
            )
            # check correct parameters passed
            plan = PLAN.copy()
            for node, states in c.start.items():
                ref_states = plan.pop(node.key)
                ref_states = [ref_states] * 3 if isinstance(ref_states, float) else ref_states
                self.assertListEqual(ref_states, list(states), msg=f"Wrong start states passed for node {node}")
        except ApplicationError:
            pass
