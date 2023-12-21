import logging

import numpy as np
import pyomo.environ as pyo

logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class Plant:
    """Encapsulates the mathematical model representing the power plant: topology (nodes and edges), costs and demands."""

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.costs = {}
        self.demands = {}
        self.model = pyo.ConcreteModel()

    def add_node(self, node):
        """Add a node to the plant topology and declare the variables and constraints that mathematically describe it.

        Args:
            node : a Node object
        """

        node._add_to_plant(self)
        self.nodes.append((node.name, node.node_type))

    def add_edge(self, node_out: str, node_in: str, commodity: str, lb: float = 0., ub: float = np.inf):
        """Add an edge to the plant topology.

        Args:
            node_out : the source node
            node_in : the destination node
            commodity: the commodity flowing through the edge
            lb : the minimum flow supported by the edge
            ub : the maximum flow supported by the edge
        """

        edge = (node_out, node_in, commodity, lb, ub)
        self.edges.append(edge)

    def expand_network(self):
        """Declare the variables and constraints that mathematically describe the connectivity of the plant"""

        def bounds(model, node_out, node_in, commodity, lb, ub):
            return (lb, ub)

        self.model.flows = pyo.Var(self.edges, domain=pyo.NonNegativeReals, bounds=bounds)

        for node, _ in self.nodes:
            node = self.model.component(node)

            @node.Constraint(node.gates_in_index)
            def flows_in(node, commodity):
                return node.gates_in[commodity] == sum(
                    self.model.flows[(node_out, node_in, com, lb, ub)] for (node_out, node_in, com, lb, ub) in
                    self.edges if node_in == node.name and com == commodity)

            @node.Constraint(node.gates_out_index)
            def flows_out(node, commodity):
                return node.gates_out[commodity] == sum(
                    self.model.flows[(node_out, node_in, com, lb, ub)] for (node_out, node_in, com, lb, ub) in
                    self.edges if node_out == node.name and com == commodity)

    def add_predicted_plan(self, flows: dict[tuple[str, str, str, float, float], float],
                           machines: dict[str, tuple[bool, float]]):
        """Declare the variables and constraints that describe the predicted plan, from which the recourse action generates a new plan

        Args:
            flows : a dictionary {(node_in, node_out, commodity, lb, ub) : flow} indicating the predicted flow for each edge
            machine: a dictionary {machine : (switch, setpoint)} indicating the predicted state for each machine
        """

        self.model.d_flows = pyo.Var(self.edges, domain=pyo.NonNegativeReals)

        @self.model.Constraint(self.edges)
        def flows_distance_geq(model, node_out, node_in, commodity, lb, ub):
            edge = (node_out, node_in, commodity, lb, ub)
            return model.d_flows[edge] >= model.flows[edge] - flows[edge]

        @self.model.Constraint(self.edges)
        def flows_distance_leq(model, node_out, node_in, commodity, lb, ub):
            edge = (node_out, node_in, commodity, lb, ub)
            return model.d_flows[edge] >= flows[edge] - model.flows[edge]

        self.model.d_setpoints = pyo.Var(machines.keys(), domain=pyo.NonNegativeReals)
        self.model.switchoffs = pyo.Var(machines.keys(), domain=pyo.Binary)
        self.model.switchons = pyo.Var(machines.keys(), domain=pyo.Binary)

        @self.model.Constraint(machines)
        def switchon(model, machine):
            node = model.component(machine)
            return model.switchons[machine] >= node.switch - machines[machine][0]

        @self.model.Constraint(machines)
        def switchoff(model, machine):
            node = model.component(machine)
            return model.switchoffs[machine] >= machines[machine][0] - node.switch

        @self.model.Constraint(machines)
        def setpoints_distance_geq(model, machine):
            node = model.component(machine)
            return model.d_setpoints[machine] >= node.setpoint - machines[machine][1]

        @self.model.Constraint(machines)
        def setpoints_distance_leq(model, machine):
            node = model.component(machine)
            return model.d_setpoints[machine] >= machines[machine][1] - node.setpoint

    def add_objective(self, cost_weight: float = 1, machine_weight: dict[str, tuple[float, float, float]] | float = 1):
        """Declare the objective function

        Args:
            cost_weight : the objective coefficient for the total cost of the produced plan 
            machine_weight: either a float used as the objective coefficient for the total difference between the predicted and the produced plan in terms of machine states, or a dictionary 
                            {machine : (switchon, switchoff, setpoint)} containing the objective coefficients of the switchon, switchoff and setpoint difference for each individual machine
        """

        expr = 0

        if cost_weight:
            expr += cost_weight * sum(self.costs[node, commodity] * self.model.component(node).gates_out[commodity] for
                                      (node, commodity), cost in self.costs.items() if cost >= 0)
            expr += cost_weight * sum(
                self.costs[node, commodity] * self.model.component(node).gates_in[commodity] for (node, commodity), cost
                in self.costs.items() if cost < 0)

        if machine_weight:
            if type(machine_weight) != dict:
                machine_weight = {node: (machine_weight, machine_weight, machine_weight) for node, node_type in
                                  self.nodes if node_type == 'machine'}

            expr += sum(
                machine_weight[node][0] * self.model.switchoffs[node] + machine_weight[node][1] * self.model.switchons[
                    node] for node, node_type in self.nodes if node_type == 'machine')
            expr += sum(machine_weight[node][2] * self.model.d_setpoints[node] for node, node_type in self.nodes if
                        node_type == 'machine')

        self.model.objective = pyo.Objective(expr=expr, sense=pyo.minimize)

    def write(self, filename, node=None):
        self.model.write(filename, io_options={"symbolic_solver_labels": True})


class Node:
    """A generic node with all variables and constraints mathematically describing it"""

    def __init__(self, name: str, gates_in: list[str], gates_out: list[str]):
        """
        Args: 
            name : name of the node
            gates_in : a list of input commodities for the node
            gates_out : a list of output commodities for the node
        """

        self.name = name
        self.gates_in = gates_in
        self.gates_out = gates_out
        self.node_type = 'generic'

    def _add_to_plant(self, plant: Plant):
        # Block
        self.node = pyo.Block(concrete=True)

        # Variables
        self.node.gates_in = pyo.Var(self.gates_in, domain=pyo.NonNegativeReals)
        self.node.gates_out = pyo.Var(self.gates_out, domain=pyo.NonNegativeReals)

        plant.model.add_component(name=self.name, val=self.node)


class Supplier(Node):
    """A supplier node with all variables and constraints mathematically describing it"""

    def __init__(self, name: str, costs: dict[str, float]):
        """
        Args: 
            name : name of the node
            costs : a dictionary {commodity: cost} indicating the price requested by the supplier for each sold commodity
        """

        super().__init__(name, gates_in=[], gates_out=list(costs.keys()))
        self.costs = costs
        self.node_type = 'supplier'

    def _add_to_plant(self, plant: Plant):
        # Block
        self.node = pyo.Block(concrete=True)

        # Variables
        self.node.gates_in = pyo.Var(self.gates_in, domain=pyo.NonNegativeReals)
        self.node.gates_out = pyo.Var(self.gates_out, domain=pyo.NonNegativeReals)

        plant.model.add_component(name=self.name, val=self.node)
        plant.costs.update({(self.name, commodity): self.costs[commodity] for commodity in self.costs.keys()})


class Purchaser(Node):
    """A purchaser node with all variables and constraints mathematically describing it"""

    def __init__(self, name: str, profits: dict[str, float]):
        """
        Args: 
            name : name of the node
            costs : a dictionary {commodity : price} indicating the price paid by the purchaser for each bought commodity
        """

        super().__init__(name, gates_in=list(profits.keys()), gates_out=[])
        self.profits = profits
        self.node_type = 'grid'

    def _add_to_plant(self, plant: Plant):
        # Block
        self.node = pyo.Block(concrete=True)

        # Variables
        self.node.gates_in = pyo.Var(self.gates_in, domain=pyo.NonNegativeReals)
        self.node.gates_out = pyo.Var(self.gates_out, domain=pyo.NonNegativeReals)

        plant.model.add_component(name=self.name, val=self.node)
        plant.costs.update({(self.name, commodity): -self.profits[commodity] for commodity in self.profits.keys()})


class Consumer(Node):
    """A consumer node with all variables and constraints mathematically describing it"""

    def __init__(self, name: str, demands: dict[str, float]):
        """
        Args: 
            name : name of the node
            costs : a dictionary {commodity : demand} indicating the demand of the consumer for each required commodity
        """

        super().__init__(name, gates_in=list(demands.keys()), gates_out=[])
        self.demands = demands
        self.node_type = 'consumer'

    def _add_to_plant(self, plant: Plant):
        # Block
        self.node = pyo.Block(concrete=True)

        # Variables
        self.node.gates_in = pyo.Var(self.gates_in, domain=pyo.NonNegativeReals)
        self.node.gates_out = pyo.Var(self.gates_out, domain=pyo.NonNegativeReals)

        # Constraints
        @self.node.Constraint(self.gates_in)
        def demands(node, commodity):
            return node.gates_in[commodity] == self.demands[commodity]

        plant.model.add_component(name=self.name, val=self.node)
        plant.demands.update({(self.name, commodity): self.demands[commodity] for commodity in self.demands.keys()})


class Routing(Node):
    """A routing node with all variables and constraints mathematically describing it"""

    def __init__(self, name: str, commodities: list[str]):
        """
        Args: 
            name : name of the node
            commodities : a list of the commodities flowing through the node
        """

        super().__init__(name, gates_in=commodities, gates_out=commodities)
        self.commodities = commodities
        self.node_type = 'routing'

    def _add_to_plant(self, plant: Plant):
        # Block
        self.node = pyo.Block(concrete=True)

        # Variables
        self.node.gates_in = pyo.Var(self.gates_in, domain=pyo.NonNegativeReals)
        self.node.gates_out = pyo.Var(self.gates_out, domain=pyo.NonNegativeReals)

        # Constraints
        @self.node.Constraint(self.commodities)
        def flow_conservation(node, commodity):
            return node.gates_in[commodity] == node.gates_out[commodity]

        plant.model.add_component(name=self.name, val=self.node)


class Storage(Node):
    """A storage node with all variables and constraints mathematically describing it"""

    def __init__(self, name, storage: dict[str, tuple[float, float, float, float]]):
        """
        Args: 
            name : name of the node
            storage : a dictionary {commodity : (storage, capacity, charge_rate, discharge_rate)} indicating the actual and maximum amount, 
                      and the charge and discharge rate for each storable commodity
        """

        super().__init__(name, gates_in=storage.keys(), gates_out=storage.keys())
        self.storage = storage
        self.node_type = 'storage'

    def _add_to_plant(self, plant):
        # Block
        self.node = pyo.Block(concrete=True)

        # Variables
        def capacity(model, commodity):
            capacity = self.storage[commodity][1]
            return (0, capacity)

        def charge_rate(model, commodity):
            rate = self.storage[commodity][2]
            return (0, rate)

        def discharge_rate(model, commodity):
            rate = self.storage[commodity][3]
            return (0, rate)

        self.node.gates_in = pyo.Var(self.gates_in, domain=pyo.NonNegativeReals, bounds=charge_rate)
        self.node.gates_out = pyo.Var(self.gates_out, domain=pyo.NonNegativeReals, bounds=discharge_rate)
        self.node.storage = pyo.Var(self.storage.keys(), domain=pyo.NonNegativeReals, bounds=capacity)

        # Constraints
        @self.node.Constraint(self.storage.keys())
        def storage_selection(node, commodity):
            return node.storage[commodity] == node.gates_in[commodity] - node.gates_out[commodity] + \
                   self.storage[commodity][0]

        plant.model.add_component(name=self.name, val=self.node)


class DMachine(Node):
    """A discrete machine node, modeled as a piecewise linear function, with all variables and constraints mathematically describing it"""

    def __init__(self, name,
                 setpoints: list[float], values_in: dict[str, list[float]], values_out: dict[str, list[float]]):
        """
        Args: 
            name : name of the node
            setpoints : a list of the setpoint breaks of the picewise linear function, with the setpoint representing the discrete independent variable of the function  
            values_in : a list of the input commodities breaks of the picewise linear function, with each input commodity representing a dependent variable of the function  
            values_in : a list of the output commodities breaks of the picewise linear function, with each output commodity representing a dependent variable of the function  
        """

        super().__init__(name, gates_in=list(values_in.keys()), gates_out=list(values_out.keys()))
        self.setpoints = setpoints
        self.values_in = values_in
        self.values_out = values_out
        self.node_type = 'machine'

    def _add_to_plant(self, plant):
        # Block
        self.node = pyo.Block(concrete=True)

        # Variables
        self.node.gates_in = pyo.Var(self.gates_in, domain=pyo.NonNegativeReals)
        self.node.gates_out = pyo.Var(self.gates_out, domain=pyo.NonNegativeReals)
        self.node.b = pyo.Var(range(len(self.setpoints)), domain=pyo.Binary)
        self.node.switch = pyo.Var(domain=pyo.Binary)
        self.node.setpoint = pyo.Var(domain=pyo.Integers)

        # Constraints
        self.node.b_selection = pyo.Constraint(
            expr=sum(self.node.b[k] for k in range(len(self.setpoints))) == self.node.switch)
        self.node.setpoint_selection = pyo.Constraint(
            expr=self.node.setpoint == sum(self.setpoints[k] * self.node.b[k] for k in range(len(self.setpoints))))

        @self.node.Constraint(self.gates_in)
        def gates_in_selection(node, commodity):
            return node.gates_in[commodity] == sum(
                self.values_in[commodity][k] * node.b[k] for k in range(len(self.setpoints)))

        @self.node.Constraint(self.gates_out)
        def gates_out_selection(node, commodity):
            return node.gates_out[commodity] == sum(
                self.values_out[commodity][k] * node.b[k] for k in range(len(self.setpoints)))

        plant.model.add_component(name=self.name, val=self.node)


class CMachine(Node):
    """A continuos machine node, modeled as a piecewise linear function, with all variables and constraints mathematically describing it"""

    def __init__(self, name,
                 setpoints: list[float], values_in: dict[str, list[float]], values_out: dict[str, list[float]]):
        """
        Args: 
            name : name of the node
            setpoints : a list of the setpoint breaks of the picewise linear function, with the setpoint representing the continuous independent variable of the function  
            values_in : a list of the input commodities breaks of the picewise linear function, with each input commodity representing a dependent variable of the function  
            values_in : a list of the output commodities breaks of the picewise linear function, with each output commodity representing a dependent variable of the function  
        """

        super().__init__(name, gates_in=list(values_in.keys()), gates_out=list(values_out.keys()))
        self.setpoints = setpoints
        self.values_in = values_in
        self.values_out = values_out
        self.node_type = 'machine'

    def _add_to_plant(self, plant):
        # Block
        self.node = pyo.Block(concrete=True)

        # Variables
        self.node.gates_in = pyo.Var(self.gates_in, domain=pyo.NonNegativeReals)
        self.node.gates_out = pyo.Var(self.gates_out, domain=pyo.NonNegativeReals)
        self.node.b = pyo.Var(range(len(self.setpoints)), domain=pyo.Binary)
        self.node.alpha = pyo.Var(range(len(self.setpoints)), domain=pyo.NonNegativeReals)
        self.node.switch = pyo.Var(domain=pyo.Binary)
        self.node.setpoint = pyo.Var(domain=pyo.Reals)

        # pyo.Constraints
        self.node.setpoint_selection = pyo.Constraint(
            expr=self.node.setpoint == sum(self.setpoints[k] * self.node.alpha[k] for k in range(len(self.setpoints))))

        @self.node.Constraint(self.gates_in)
        def gates_in_selection(node, commodity):
            return node.gates_in[commodity] == sum(
                self.values_in[commodity][k] * node.alpha[k] for k in range(len(self.setpoints)))

        @self.node.Constraint(self.gates_out)
        def gates_out_selection(node, commodity):
            return node.gates_out[commodity] == sum(
                self.values_out[commodity][k] * node.alpha[k] for k in range(len(self.setpoints)))

        self.node.alpha_selection = pyo.Constraint(
            expr=sum(self.node.alpha[k] for k in range(len(self.setpoints))) == self.node.switch)

        @self.node.Constraint(range(len(self.setpoints)))
        def alpha_ub(node, k):
            return node.alpha[k] <= node.b[k]

        self.node.b_selection = pyo.Constraint(
            expr=sum(self.node.b[k] for k in range(len(self.setpoints))) <= 2 * self.node.switch)
        self.node.b_init_consecutve = pyo.Constraint(expr=self.node.b[0] <= self.node.b[1])
        self.node.b_last_consecutve = pyo.Constraint(
            expr=self.node.b[range(len(self.setpoints))[-1]] <= self.node.b[range(len(self.setpoints))[-2]])

        @self.node.Constraint(range(len(self.setpoints))[1:-1])
        def b_consecutive(node, k):
            return node.b[k] <= node.b[k - 1] + node.b[k + 1]

        plant.model.add_component(name=self.name, val=self.node)
