import unittest

from ppsim import Plant

PLANT = Plant(horizon=3)
PLANT.add_supplier(name='sup', commodity='in', predictions=1.)
PLANT.add_machine(name='mac_1', parents='sup', commodity='in', setpoint={'setpoint': [1., 3.], 'out': [1., 3.]})
PLANT.add_machine(name='mac_2', parents='sup', commodity='in', setpoint={'setpoint': [1., 3.], 'out': [1., 3.]})
PLANT.add_storage(name='sto', parents='mac_1', commodity='out')
PLANT.add_client(name='cus', parents=['sto', 'mac_1'], commodity='out', predictions=1.)
PLANT.add_client(name='pur', parents=['sto', 'mac_2'], commodity='out', predictions=1., purchaser=True)

PLAN = {
    ('sup', 'mac_1'): [1., 2., 3.],
    ('sup', 'mac_2'): [1., 2., 3.],
    ('mac_1', 'cus'): [1., 2., 3.],
    ('mac_1', 'sto'): [1., 2., 3.],
    ('mac_2', 'pur'): [1., 2., 3.],
    ('sto', 'cus'): [1., 2., 3.],
    ('sto', 'pur'): [1., 2., 3.]
}

FLOWS = {
    ('sup', 'mac_1'): [1., 2., 3.],
    ('sup', 'mac_2'): [0., 0., 0.],
    ('mac_1', 'cus'): [1., 2., 3.],
    ('mac_1', 'sto'): [0., 0., 0.],
    ('mac_2', 'pur'): [0., 0., 0.],
    ('sto', 'cus'): [0., 0., 0.],
    ('sto', 'pur'): [0., 0., 0.]
}


def recourse_action(step, _):
    return {edge: flows[step] for edge, flows in FLOWS.items()}


class TestPlantRun(unittest.TestCase):
    def test_correct_behaviour(self):
        pass
