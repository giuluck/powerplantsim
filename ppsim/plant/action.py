from abc import abstractmethod
from typing import Dict, Tuple, Union, Optional, Callable, Any

import numpy as np
import pyomo.environ as pyo

from ppsim.utils.typing import Plan


class RecourseAction:
    """Abstract class that defines the recourse action to apply to the predicted plan to generate the actual plan,
    after the true costs and demands are disclosed."""

    def __init__(self):
        self._plant: Optional = None

    def build(self, plant):
        """Builds the recourse action by attaching it to a specific plant.

        :param plant:
            The power plant object.

        :return:
            The RecourseAction instance itself.
        """
        assert self._plant is None, "Recourse action already built."
        self._plant = plant
        return self

    @abstractmethod
    def execute(self) -> Plan:
        """Performs the recourse action on the power plant for a given step.

        :return:
            A dictionary {machine | edge: <updated_state | updated_flow>} where a machine is identified by its name and
            an edge is identified by the tuple of the names of the node that is connecting, while updated_state and
            updated_flow is the value of the actual state/flow.
        """
        pass


class CallableRecourseAction(RecourseAction):
    """A recourse action which is built from a function f(plant) -> plan."""

    def __init__(self, action: Callable[[Any], Plan]):
        """
        :param action:
            A function f(plant) -> plan which performs the recourse action on the power plant for a given step.
        """
        super(CallableRecourseAction, self).__init__()
        self._action: Callable[[Any], Plan] = action

    def execute(self) -> Plan:
        return self._action(self._plant)


class DefaultRecourseAction(RecourseAction):
    """A default, greedy implementation of the recourse action."""

    MachineWeight = Tuple[float, float, float]
    """Type to define the cost of operating a machine, i.e., a tuple (switch_on, switch_off, state)."""

    def __init__(self,
                 solver: str,
                 decimals: int = 9,
                 cost_weight: Optional[float] = 1.0,
                 storage_weight: Union[None, float, Dict[str, float]] = 1.0,
                 machine_weight: Union[None, MachineWeight, Dict[str, MachineWeight]] = (1.0, 1.0, 0.0)):
        """
        :param solver:
            The underlying MIP solver to use in Pyomo.

        :param decimals:
            The decimals used for final rounding operations to account for numerical errors in the solving process.

        :param cost_weight:
            The objective coefficient for the total cost of the produced plan.

        :param storage_weight:
            Either a float used as the objective coefficient for the total difference between the predicted and the
            produced plan in terms of storage states, or a dictionary {storage : accumulated_storage_diff} containing
            the objective coefficients of the accumulated storage difference for individual storages.

        :param machine_weight:
            Either a float used as the objective coefficient of the total difference between the predicted and the
            produced plan in terms of machine states, or a dictionary {machine : (switch_on, switch_off, state)}
            containing the objective coefficients of the switch_on, switch_off and state difference for individual
            machines.
        """
        super(DefaultRecourseAction, self).__init__()

        self._solver: str = solver
        self._decimals: int = decimals
        self._cost_weight: Optional[float] = cost_weight
        self._storage_weight: Union[None, float, Dict[str, float]] = storage_weight
        self._machine_weight: Union[None, float, Dict[str, DefaultRecourseAction.MachineWeight]] = machine_weight

    def build(self, plant):
        super(DefaultRecourseAction, self).build(plant)
        # handle the storage weight input in case a single value is passed instead of dictionary
        if isinstance(self._storage_weight, float):
            self._storage_weight = {s: self._storage_weight for s in self._plant.storages}
        # handle the machine weight input in case a single value is passed instead of dictionary
        if isinstance(self._machine_weight, tuple):
            self._machine_weight = {m: self._machine_weight for m in self._plant.machines}
        return self

    # noinspection PyTypeChecker
    def execute(self) -> Plan:
        # retrieve the pyomo model representing the plant and define a variable for the objective function
        model = self._plant.to_pyomo(mutable=False)
        objective = 0.0
        # for each node with a price (i.e., suppliers and purchasers) add the cost to the objective function
        if self._cost_weight is not None:
            for cmp in [*self._plant.suppliers, *self._plant.purchasers, *self._plant.machines]:
                objective += self._cost_weight * model.component(cmp).cost
        # for each storage, add the cost for storage difference to the objective function
        if self._storage_weight is not None:
            for storage, weight in self._storage_weight.items():
                cmp = model.component(storage)
                # model a variable for the storage difference (i.e., | storage - current_storage |)
                #  - the absolute value is relaxed to storage_diff >= | storage - current_storage |
                #  - this relaxation allows to linearize the constraint into two different ones
                #  - eventually, the value is multiplied by the respective weight and added to the objective
                cmp.storage_diff = pyo.Var(domain=pyo.NonNegativeReals, initialize=0.0)
                cmp.storage_diff_geq = pyo.Constraint(rule=cmp.storage_diff >= cmp.storage - cmp.current_storage)
                cmp.storage_diff_leq = pyo.Constraint(rule=cmp.storage_diff >= cmp.current_storage - cmp.storage)
                objective += weight * cmp.storage_diff
        # for each machine, add the costs for on/off/state to the objective function
        if self._machine_weight is not None:
            for machine, (on_weight, off_weight, state_weight) in self._machine_weight.items():
                cmp = model.component(machine)
                mac = self._plant.machines[machine]
                # define variables for switch change (use variables instead of expressions to avoid pyomo errors)
                #  - on_diff == 1 if the machine is on (cmp.switch == 1) and it was set as off (was_on == 0)
                #  - off_diff == 1 if the machine is off (cmp.switch == 0) and it was set as on (was_on == 1)
                was_on = 0 if np.isnan(pyo.value(mac.current_state)) else 1
                cmp.on_diff = pyo.Var(domain=pyo.Binary, initialize=0)
                cmp.on_diff_cst = pyo.Constraint(rule=cmp.on_diff == (1 - was_on) * cmp.switch)
                cmp.off_diff = pyo.Var(domain=pyo.Binary, initialize=0)
                cmp.off_diff_cst = pyo.Constraint(rule=cmp.off_diff == was_on * (1 - cmp.switch))
                # define a variable for state change, i.e. state_diff == | node.current_state - node.state |
                #  - this value has a meaning only if the machine was set as on, and it is still on
                #  - otherwise, the on_diff and off_diff variables are considered in the objective
                # in order to model the state difference when the machine is off we rely on the big-M formulation
                #  - M is computed as the maximal setpoint difference, i.e., the largest state
                #  - additionally, we define a binary control variable z = was_on * switch
                #  - using z, we force state_diff <= 0 whenever z == 0, i.e., either the machine was set or is off
                #  - then, we add the constraint state_diff >= | current_state - state | with the term -z * gap
                #  - this last term is used to guarantee that the rhs is negative (i.e, trivial) whenever z == 0
                z = was_on * cmp.switch
                m = mac.setpoint.index[-1]
                cmp.state_diff = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, m), initialize=0.0)
                cmp.state_diff_m = pyo.Constraint(rule=cmp.state_diff <= z * m)
                cmp.state_diff_geq = pyo.Constraint(rule=cmp.state_diff >= cmp.state - cmp.current_state - (1 - z) * m)
                cmp.state_diff_leq = pyo.Constraint(rule=cmp.state_diff >= cmp.current_state - cmp.state - (1 - z) * m)
                # eventually, multiply each value by the respective weight and add them to the objective
                objective += on_weight * cmp.on_diff + off_weight * cmp.off_diff + state_weight * cmp.state_diff
        # add the objective function to the model, solve it using the defined solver, and eventually return the plan
        model.objective = pyo.Objective(expr=objective, sense=pyo.minimize)
        pyo.SolverFactory(self._solver).solve(model)
        plan = {}
        for edge in self._plant.edges().values():
            cmp = model.component(edge.name)
            plan[edge.simple_key] = self._get_value(cmp.flow)
        for machine in self._plant.machines.values():
            cmp = model.component(machine.name)
            plan[machine.key] = self._get_value(cmp.state) if self._get_value(cmp.switch) == 1 else np.nan
        return plan

    def _get_value(self, variable) -> float:
        return np.round(pyo.value(variable), decimals=self._decimals)
