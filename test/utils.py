# test plant with step = 0 to simulate first time step
from typing import Optional

from ppsim import Plant


class TestPlant(Plant):
    @property
    def step(self) -> Optional[int]:
        return 0


SOLVER = 'gurobi'
