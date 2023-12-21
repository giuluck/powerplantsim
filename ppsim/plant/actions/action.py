from abc import abstractmethod
from typing import Dict, Tuple, Union, Optional

import numpy as np
import pyomo.environ as pyo

from ppsim.plant.actions.plant import Supplier, CMachine, DMachine, Storage, Purchaser, Consumer, Plant
from ppsim.utils.typing import Plan


class RecourseAction:
    """Abstract class that defines the recourse action to apply to the predicted plan to generate the actual plan,
    after the true costs and demands are disclosed."""

    def __init__(self):
        self.plant: Plant = Plant()
        self._plant: Optional = None
        """The Plant object which is attached to the recourse action."""

    def build(self, plant):
        """Builds the recourse action by attaching it to a specific plant.

        :param plant:
        """
        assert self._plant is None, "Recourse action already built."
        self._plant = plant

    @abstractmethod
    def execute(self) -> Plan:
        """Performs the recourse action on the power plant for a given step.

        :return:
            A dictionary {machine | edge: <updated_state | updated_flow>} where a machine is identified by its name and
            an edge is identified by the tuple of the names of the node that is connecting, while updated_state and
            updated_flow is the value of the actual state/flow.
        """
        pass


class DefaultRecourseAction(RecourseAction):
    """A default, greedy implementation of the recourse action."""

    def __init__(self,
                 solver: str = 'gurobi',
                 cost_weight: float = 1.0,
                 machine_weight: Union[float, Dict[str, Tuple[float, float, float]]] = 1.0,
                 storage_weight: Union[float, Dict[Tuple[str, str], float]] = 1.0):
        """
        :param solver:
            The underlying MIP solver to use in Pyomo.

        :param cost_weight:
            The objective coefficient for the total cost of the produced plan.

        :param machine_weight:
            Either a float used as the objective coefficient of the total difference between the predicted and the
            produced plan in terms of machine states, or a dictionary {machine : (switch_on, switch_off, setpoint)}
            containing the objective coefficients of the switch_on, switch_off and setpoint difference for individual
            machines.

        :param storage_weight:
            Either a float used as the objective coefficient for the total difference between the predicted and the
            produced plan in terms of storage states, or a dictionary {(storage, commodity) : accumulated_storage_diff}
            containing the objective coefficients of the accumulated storage difference for individual storages.
        """
        super(DefaultRecourseAction, self).__init__()

        self.solver = solver
        """The underlying MIP solver to use in Pyomo."""

        self.cost_weight = cost_weight
        """The objective coefficient for the total cost of the produced plan."""

        self.machine_weight: Union[float, Dict[str, Tuple[float, float, float]]] = machine_weight
        """Either a float used as the objective coefficient of the total difference between the predicted and the
        produced plan in terms of machine states, or a dictionary {machine : (switch_on, switch_off, setpoint)}
        containing the objective coefficients of the switch_on, switch_off and setpoint difference for individual
        machines."""

        self.storage_weight: Union[float, Dict[Tuple[str, str], float]] = storage_weight
        """Either a float used as the objective coefficient for the total difference between the predicted and the
        produced plan in terms of storage states, or a dictionary {(storage, commodity) : accumulated_storage_diff}
        containing the objective coefficients of the accumulated storage difference for individual storages."""

    def _build(self):
        """Builds the mathematical model representing the power plant"""

        self.predicted_flows = {}
        self.predicted_machines = {}
        self.predicted_storages = {}

        # Nodes
        nodes = []
        for supplier in self._plant.suppliers.values():
            unit = Supplier(name=supplier.name, costs={supplier.commodity: supplier.current_value})
            nodes.append(unit)
        for machine in self._plant.machines.values():
            if np.isnan(machine.current_state):
                self.predicted_machines[machine.name] = (0., 0.)
            else:
                self.predicted_machines[machine.name] = (1., machine.current_state)
            if machine.discrete_setpoint:
                unit = DMachine(
                    name=machine.name,
                    setpoints=machine.setpoint.index.tolist(),
                    values_in={commodity: values.tolist() for commodity, values in machine.setpoint['input'].items()},
                    values_out={commodity: values.tolist() for commodity, values in machine.setpoint['output'].items()}
                )
            else:
                unit = CMachine(
                    name=machine.name,
                    setpoints=machine.setpoint.index.tolist(),
                    values_in={commodity: values.tolist() for commodity, values in machine.setpoint['input'].items()},
                    values_out={commodity: values.tolist() for commodity, values in machine.setpoint['output'].items()}
                )
            nodes.append(unit)
        for storage in self._plant.storages.values():
            self.predicted_storages = {(storage.name, storage.commodity): storage.current_storage}
            unit = Storage(name=storage.name, storage={storage.commodity: (
                storage.current_storage,
                storage.capacity,
                storage.charge_rate,
                storage.discharge_rate
            )})
            nodes.append(unit)
        for client in self._plant.clients.values():
            if client.purchaser:
                unit = Purchaser(name=client.name, profits={client.commodity: client.current_value})
            else:
                unit = Consumer(name=client.name, demands={client.commodity: client.current_value})
            nodes.append(unit)

        for unit in nodes:
            self.plant.add_node(unit)

        # Edges
        for edge in self._plant.edges().values():
            key = (edge.source, edge.destination, edge.commodity, edge.min_flow, edge.max_flow)
            self.predicted_flows[key] = edge.current_flow
            self.plant.add_edge(
                node_out=edge.source,
                node_in=edge.destination,
                commodity=edge.commodity,
                lb=edge.min_flow,
                ub=edge.max_flow
            )

        self.plant.expand_network()

        # Predicted Plan
        self.plant.add_predicted_plan(
            flows=self.predicted_flows,
            machines=self.predicted_machines,
            storages=self.predicted_storages
        )

    def _solve(self):
        """Solves the mathematical model to obtain a new feasible plan, starting from the predicted one."""
        self.plant.add_objective(
            cost_weight=self.cost_weight,
            machine_weight=self.machine_weight,
            storage_weight=self.storage_weight
        )
        solver = pyo.SolverFactory(self.solver)
        results = solver.solve(self.plant.model)
        self.termination = results.solver.termination_condition

    def execute(self) -> Plan:
        self.plant = Plant()
        self._build()
        self._solve()

        if self.termination == 'optimal':
            flows = {edge: self.plant.model.flows[edge].value for edge in self.plant.edges}
            machines = {}
            storages = {}
            for node, node_type in self.plant.nodes:
                if node_type == 'machine':
                    machines[node] = (
                        self.plant.model.component(node).switch.value, self.plant.model.component(node).setpoint.value)
                elif node_type == 'storage':
                    for commodity in self.plant.model.component(node).storage_index.data():
                        storages[(node, commodity)] = self.plant.model.component(node).storage[commodity].value

            return {
                **{k: v[1] if v[0] == 1 else np.nan for k, v in machines.items()},
                **{(k[0], k[1]): v for k, v in flows.items()}
            }
        else:
            raise StopIteration(self.termination)
