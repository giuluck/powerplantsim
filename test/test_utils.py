# test plant with step = 0 to simulate first time step
from typing import Optional

import numpy as np

from powerplantsim import Plant


class DummyPlant(Plant):
    @property
    def step(self) -> Optional[int]:
        return 0


SOLVER = 'glpk'

SETPOINT = dict(commodity='in', setpoint=[1., 3.], inputs=[1., 3.], outputs={'out': [1., 3.]})

PLANT = Plant(horizon=3)
PLANT.add_extremity(kind='supplier', name='sup', commodity='in', predictions=[1., 2., 3.])
PLANT.add_machine(name='mac_1', parents='sup', discrete=False, **SETPOINT)
PLANT.add_machine(name='mac_2', parents='sup', discrete=True, **SETPOINT)
PLANT.add_storage(name='sto', parents='mac_1', commodity='out', capacity=100)
PLANT.add_extremity(kind='customer', name='cus', parents=['sto', 'mac_1'], commodity='out', predictions=[1., 2., 3.])
PLANT.add_extremity(kind='purchaser', name='pur', parents=['sto', 'mac_2'], commodity='out', predictions=[1., 2., 3.])

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
