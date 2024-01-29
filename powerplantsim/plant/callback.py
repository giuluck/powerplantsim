from abc import abstractmethod
from typing import Dict

import pandas as pd

from powerplantsim.datatypes import Edge, Node
from powerplantsim.utils.typing import State, Flow


class Callback:
    """Basic callback interface for a simulation."""

    @abstractmethod
    def on_simulation_start(self, plant, states: Dict[Node, pd.Series], flows: Dict[Edge, pd.Series]):
        """Callback hook that is called before the first step of the simulation.

        :param plant:
            The plant object on which the callback is attached.

        :param states:
            The input states of the whole plan indexed by node.

        :param flows:
            The input flows of the whole plan indexed by edge.
        """
        pass

    @abstractmethod
    def on_iteration_update(self, plant):
        """Callback hook that is called soon after the datatypes are updated with the given plan.

        :param plant:
            The plant object on which the callback is attached.
        """
        pass

    @abstractmethod
    def on_iteration_recourse(self, plant, states: Dict[Node, State], flows: Dict[Edge, Flow]):
        """Callback hook that is called soon after the actual plan is computed by the recourse action.

        :param plant:
            The plant object on which the callback is attached.

        :param states:
            The input states of the given iteration indexed by node.

        :param flows:
            The input flows of the given iteration indexed by edge.
        """
        pass

    @abstractmethod
    def on_iteration_step(self, plant):
        """Callback hook that is called soon after the datatypes are updated with the actual plan.

        :param plant:
            The plant object on which the callback is attached.
        """
        pass

    @abstractmethod
    def on_simulation_end(self, plant):
        """Callback hook that is called once the simulation is ended.

        :param plant:
            The plant object on which the callback is attached.
        """
        pass
