import unittest

import numpy as np
import pytest

from powerplantsim import Plant
from powerplantsim.plant import DefaultRecourseAction
from test.test_utils import SOLVER, IN_GITHUB_ACTIONS

PLANT = Plant(horizon=1)
PLANT.add_extremity(kind='supplier', name='sup', commodity='in_com', predictions=1.)
PLANT.add_machine(
    name='d_mac',
    parents='sup',
    commodity='in_com',
    setpoint=[0.2, 0.5, 1.0],
    inputs=[0.1, 1.0, 2.0],
    outputs={'out_com_1': [0.2, 0.5, 0.9], 'out_com_2': [2.0, 5.0, 9.0]},
    discrete=True
)
PLANT.add_machine(
    name='c_mac',
    parents='sup',
    commodity='in_com',
    setpoint=[0.2, 0.5, 1.0],
    inputs=[0.2, 2.0, 4.0],
    outputs={'out_com_1': [2.0, 5.0, 9.0], 'out_com_2': [0.2, 0.5, 0.9]},
    discrete=False
)
PLANT.add_storage(name='sto', parents=['d_mac', 'c_mac'], commodity='out_com_2', capacity=100)
PLANT.add_extremity(kind='customer', name='cus', commodity='out_com_1', parents=['d_mac', 'c_mac'], predictions=6.)
PLANT.add_extremity(kind='purchaser', name='pur', commodity='out_com_2', parents=['sto'], predictions=3.)

STATES = {
    'd_mac': None,
    'c_mac': 0.625
}

FLOWS = {
    ('sup', 'c_mac'): 2.5,
    ('sup', 'd_mac'): 0.0,
    ('c_mac', 'sto'): 0.6,
    ('d_mac', 'sto'): 0.0,
    ('c_mac', 'cus'): 6.0,
    ('d_mac', 'cus'): 0.0,
    ('sto', 'pur'): 0.0
}

STORAGES = {
    'sto': 0.6
}

DEMANDS = {
    'cus': 6.0
}

BUYING_PRICES = {
    'pur': 3.0
}

SELL_PRICES = {
    'sup': 1.0
}


class TestExample(unittest.TestCase):
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Solver is absent in Github Actions")
    def test_example(self):
        p = PLANT.copy()
        output = p.run(
            plan={
                **{machine: np.nan for machine in p.machines.keys()},
                **{edge: 0.0 for edge in p.edges().keys()},
            },
            action=DefaultRecourseAction(solver=SOLVER, cost_weight=1.0, storage_weight=None, machine_weight=None),
            progress=False
        )
        for machine, state in output.states.items():
            if np.isnan(state.iloc[0]):
                self.assertEqual(
                    None,
                    STATES[machine],
                    msg=f"Wrong state returned for machine {machine} in example use case"
                )
            else:
                self.assertAlmostEqual(
                    state.iloc[0],
                    STATES[machine],
                    msg=f"Wrong state returned for machine {machine} in example use case"
                )
        for edge, flow in output.flows.items():
            self.assertAlmostEqual(
                flow.iloc[0],
                FLOWS[edge],
                msg=f"Wrong flow returned for edge {edge} in example use case"
            )
        for node, storage in output.storage.items():
            self.assertAlmostEqual(
                storage.iloc[0],
                STORAGES[node],
                msg=f"Wrong storage returned for storage {node} in example use case"
            )
        for customer, demand in output.demands.items():
            self.assertAlmostEqual(
                demand.iloc[0],
                DEMANDS[customer],
                msg=f"Wrong demand returned for customer {customer} in example use case"
            )
        for purchaser, price in output.buying_prices.items():
            self.assertAlmostEqual(
                price.iloc[0],
                BUYING_PRICES[purchaser],
                msg=f"Wrong price returned for purchaser {purchaser} in example use case"
            )
        for supplier, price in output.sell_prices.items():
            self.assertAlmostEqual(
                price.iloc[0],
                SELL_PRICES[supplier],
                msg=f"Wrong price returned for supplier {supplier} in example use case"
            )
