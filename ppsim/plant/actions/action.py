from ppsim.plant.actions.plant import *

logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class RecourseAction:
    """Defines the recourse action to apply to the predicted plan to generate the actual plan, when the true costs and demands are disclosed"""

    def __init__(self, name='recourse_action', solver='gurobi'):
        """
        name : name of the object
        solver : the underlying MIP solver to use
        """

        self.name = name
        self.solver = solver

    def _build(self, graph):
        """Builds the mathematical model representing the power plant

        Args:
            graph : a networkx object storing the power plant to model (topology, costs, demands) and the prediceted plan (flows and machine state)
        """

        self.predicted_flows = {}
        self.predicted_machines = {}
        self.predicted_storages = {}

        # Nodes
        for node in graph.nodes:
            node = graph.nodes[node]

            if node['kind'] == 'supplier':
                unit = Supplier(name=node['name'], costs={node['commodity']: node['current_price']})

            elif node['kind'] == 'machine':
                if np.isnan(node['current_state']):
                    self.predicted_machines[node['name']] = (0., 0.)
                else:
                    self.predicted_machines[node['name']] = (1., node['current_state'])

                if node['discrete_setpoint']:
                    unit = DMachine(name=node['name'],
                                    setpoints=node['setpoint'].index.tolist(),
                                    values_in={commodity: node['setpoint']['input'][commodity].tolist() for commodity in
                                               node['commodities_in']},
                                    values_out={commodity: node['setpoint']['output'][commodity].tolist() for commodity
                                                in node['commodities_out']})
                else:
                    unit = CMachine(name=node['name'],
                                    setpoints=node['setpoint'].index.tolist(),
                                    values_in={commodity: node['setpoint']['input'][commodity].tolist() for commodity in
                                               node['commodities_in']},
                                    values_out={commodity: node['setpoint']['output'][commodity].tolist() for commodity
                                                in node['commodities_out']})

            elif node['kind'] == 'storage':
                self.predicted_storages = {(node['name'], node['commodity']): node['current_storage']}
                unit = Storage(name=node['name'], storage={node['commodity']: (
                node['current_storage'], node['capacity'], node['charge_rate'], node['discharge_rate'])})

            elif node['kind'] == 'client':
                if node['purchaser']:
                    unit = Purchaser(name=node['name'], profits={node['commodity']: node['current_price']})
                else:
                    unit = Consumer(name=node['name'], demands={node['commodity']: node['current_demand']})

            self.plant.add_node(unit)

        # Edges
        for edge in graph.edges:
            edge = graph.edges[edge]
            node_out, node_in, commodity, lb, ub, flow = edge['source'], edge['destination'], edge['commodity'], edge[
                'min_flow'], edge['max_flow'], edge['current_flow']
            self.predicted_flows[(node_out, node_in, commodity, lb, ub)] = flow
            self.plant.add_edge(node_out=node_out, node_in=node_in, commodity=commodity, lb=lb, ub=ub)

        self.plant.expand_network()

        # Predicted Plan
        self.plant.add_predicted_plan(flows=self.predicted_flows, machines=self.predicted_machines,
                                      storages=self.predicted_storages)

    def _solve(self, cost_weight: float = 1, machine_weight: dict[str, tuple[float, float, float]] | float = 1,
               storage_weight: dict[tuple[str, str], float] | float = 1):
        """Solves the mathematical model to obtain a new feaibile plan, starting from the predicted one

        Args:
            cost_weight : the objective coefficient for the total cost of the produced plan
            machine_weight: either a float used as the objective coefficient of the total difference between the predicted and the produced plan in terms of machine states, or a dictionary
                            {machine : (switchon, switchoff, setpoint)} containing the objective coefficients of the switchon, switchoff and setpoint difference for individual machines
            storage_weight: either a float used as the objective coefficient for the total difference between the predicted and the produced plan in terms of storage states, or a dictionary
                            {(storage, commodity) : accumulated_storage_diff} containing the objective coefficients of the accumulated storage difference for individual storages
        """

        self.plant.add_objective(cost_weight=cost_weight, machine_weight=machine_weight, storage_weight=storage_weight)
        solver = pyo.SolverFactory(self.solver)
        results = solver.solve(self.plant.model)
        self.termination = results.solver.termination_condition

    def __call__(self, step, graph, cost_weight: float = 1,
                 machine_weight: dict[str, tuple[float, float, float]] | float = 1,
                 storage_weight: dict[tuple[str, str], float] | float = 1):
        """Builds and solves the mathematical model to obtain a new feaibile plan, starting from the predicted one

        Args:
            graph : a networkx object storing the power plant to model (topology, costs, demands) and the prediceted plan (flows and machine state)
            cost_weight : the objective coefficient for the total cost of the produced plan
            machine_weight: either a float used as the objective coefficient of the total difference between the predicted and the produced plan in terms of machine states, or a dictionary
                            {machine : (switchon, switchoff, setpoint)} containing the objective coefficients of the switchon, switchoff and setpoint difference for individual machines
            storage_weight: either a float used as the objective coefficient for the total difference between the predicted and the produced plan in terms of storage states, or a dictionary
                            {(storage, commodity) : accumulated_storage_diff} containing the objective coefficients of the accumulated storage difference for individual storages
        """

        self.plant = Plant()
        self._build(graph)
        self._solve(cost_weight=cost_weight, machine_weight=machine_weight, storage_weight=storage_weight)

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
            print(self.termination)

            return None, None, None
