import copy
from typing import Optional, Dict, Tuple, Callable, Iterable, Set, List, Any, Union, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ppsim import utils
from ppsim.datatypes import Node, Machine, Supplier, Edge, Storage, Purchaser, Customer, ExtremityNode
from ppsim.plant import drawing, execution
from ppsim.plant.action import DefaultRecourseAction, CallableRecourseAction, RecourseAction
from ppsim.plant.execution import check_plan
from ppsim.utils.typing import Setpoint, Plan


class Plant:
    """Defines a power plant based on its topology, involved commodities, and predicted prices and demands."""

    def __init__(self, horizon: Union[int, Iterable[float]], seed: int = 0):
        """
        :param horizon:
            The time horizon of the simulation.
            If an integer is passed, the index will be {0, ..., horizon - 1}.
            Otherwise, an explicit list, numpy array or pandas series can be passed.

        :param seed:
            The seed used for random number operations.
        """
        # convert horizon to a standard format (pd.Index)
        if isinstance(horizon, int):
            assert horizon > 0, f"The time horizon must be a strictly positive integer, got {horizon}"
            horizon = np.arange(horizon)
        horizon = pd.Index(horizon)

        self._rng: np.random.Generator = np.random.default_rng(seed=seed)
        self._horizon: pd.Index = horizon
        self._commodities: Set[str] = set()
        self._nodes: Dict[str, Set[Node]] = dict()
        self._edges: Set[Edge] = set()
        self._step: int = -1

    @property
    def step(self) -> Optional[int]:
        """The current step of the simulation, or None if the simulation is not started yet."""
        return None if self._step == -1 else self._step

    @property
    def index(self) -> Optional:
        """The current index of the simulation as in the time horizon, or None if the simulation is not started."""
        return None if self._step == -1 else self._horizon[self._step]

    @property
    def horizon(self) -> pd.Index:
        """A pandas series representing the time index of the simulation."""
        return self._horizon.copy()

    @property
    def commodities(self) -> Set[str]:
        return {c for c in self._commodities}

    @property
    def suppliers(self) -> Dict[str, Supplier]:
        """The supplier nodes in the plant."""
        # noinspection PyTypeChecker
        return {s.name: s for s in self._nodes.get(Supplier.kind, set())}

    @property
    def customers(self) -> Dict[str, Customer]:
        """The customer nodes in the plant."""
        # noinspection PyTypeChecker
        return {c.name: c for c in self._nodes.get(Customer.kind, set())}

    @property
    def purchasers(self) -> Dict[str, Purchaser]:
        """The purchaser nodes in the plant."""
        # noinspection PyTypeChecker
        return {c.name: c for c in self._nodes.get(Purchaser.kind, set())}

    @property
    def machines(self) -> Dict[str, Machine]:
        """The machine nodes in the plant."""
        # noinspection PyTypeChecker
        return {m.name: m for m in self._nodes.get(Machine.kind, set())}

    @property
    def storages(self) -> Dict[str, Storage]:
        """The storage nodes in the plant."""
        # noinspection PyTypeChecker
        return {s.name: s for s in self._nodes.get(Storage.kind, set())}

    def nodes(self, indexed: bool = False) -> Dict[str, Union[Node, Dict[str, Node]]]:
        """Returns the nodes in the plant.

        :param indexed:
            Whether to index the dictionary by type.

        :return:
            Either a dictionary {type: {name: node}} or a simple dictionary {name: node}.
        """
        if indexed:
            return {kind: {n.name: n for n in nodes} for kind, nodes in self._nodes.items()}
        else:
            return {n.name: n for nodes in self._nodes.values() for n in nodes}

    def edges(self,
              sources: Union[None, str, Iterable[str]] = None,
              destinations: Union[None, str, Iterable[str]] = None,
              commodities: Union[None, str, Iterable[str]] = None) -> Dict[Tuple[str, str], Edge]:
        """Returns the edges in the plant indexed either via nodes, or via nodes and commodity.

        :param sources:
            The identifier of the source nodes, used to filter just the edges starting from this node.

        :param destinations:
            The identifier of the destination nodes, used to filter just the edges ending in this node.

        :param commodities:
            The identifier of the commodities, used to filter just the edges ending in this node.

        :return:
            A dictionary <(source, destination), edge> with nodes pairs as key and edge info as value.
        """
        # get filtering functions for destinations and commodities
        check_sour = utils.get_filtering_function(user_input=sources)
        check_dest = utils.get_filtering_function(user_input=destinations)
        check_edge = utils.get_filtering_function(user_input=commodities)
        # build data structure containing all the necessary information
        return {
            (e.source, e.destination): e
            for e in self._edges
            if check_sour(e.source) and check_dest(e.destination) and check_edge(e.commodity)
        }

    def graph(self, attributes: bool = False) -> nx.DiGraph:
        """Builds the graph representing the power plant.

        :param attributes:
            Whether to include node/edge attributes.

        :return:
            A networkx DiGraph object representing the power plant.
        """
        g = nx.DiGraph()
        if attributes:
            for name, node in self.nodes().items():
                g.add_node(name, **node.dict)
            for edge in self._edges:
                g.add_edge(edge.source, edge.destination, **edge.dict)
        else:
            g.add_nodes_from(self.nodes().keys())
            g.add_edges_from(self.edges().keys())
        return g

    def to_pyomo(self, mutable: bool = False) -> pyo.ConcreteModel:
        """Encodes the plant as a Pyomo model.

        :param mutable:
            Whether to set simulation-specific parameters (e.g., current_price/current_demand/etc.) as mutable and
            without an initialization value, or to initialize them based on their specific value of the current step.

        :return:
            A pyomo.environment.ConcreteModel object modeling the plant.
        """
        # build model and define data sets for nodes and commodities
        model = pyo.ConcreteModel(name='plant')
        model.nodes = pyo.Set(initialize=[node.name for nodes in self._nodes.values() for node in nodes])
        model.commodities = pyo.Set(initialize=list(self._commodities))

        # create components for datatypes and add them to the model
        edges = {(e.source, e.destination, e.commodity): e.to_pyomo(mutable=mutable) for e in self._edges}
        nodes = {n.name: n.to_pyomo(mutable=mutable) for nodes in self._nodes.values() for n in nodes}
        for component in [*nodes.values(), *edges.values()]:
            model.add_component(component.name, component)

        # add constraints between edges variables and nodes variables
        #  - compute the input and output flows for pairs (node, commodity) as a sum of edges flows
        #  - then, use the cross product "nodes x commodities" to both the computed flow sums and the nodes' flows
        #  - finally, use these two variables (flow sums and nodes' flows) to impose an equality constraint
        in_flows = {(destination, commodity): 0.0 for _, destination, commodity in edges.keys()}
        out_flows = {(source, commodity): 0.0 for source, _, commodity in edges.keys()}
        for (source, destination, commodity), edge in edges.items():
            in_flows[(destination, commodity)] += edge.flow
            out_flows[(source, commodity)] += edge.flow

        @model.Constraint(model.nodes * model.commodities)
        def in_constraints(_, destination, commodity):
            flows = in_flows.get((destination, commodity))
            return pyo.Constraint.Skip if flows is None else flows == nodes[destination].in_flows[commodity]

        @model.Constraint(model.nodes * model.commodities)
        def out_constraints(_, source, commodity):
            flows = out_flows.get((source, commodity))
            return pyo.Constraint.Skip if flows is None else flows == nodes[source].out_flows[commodity]

        # eventually return the model
        return model

    def copy(self):
        """Copies the plant object.

        :return:
            A copy of the plant object.
        """
        return copy.deepcopy(self)

    def _check_and_update(self,
                          node: Node,
                          parents: Union[None, str, Iterable[str]],
                          commodity: Optional[str],
                          min_flow: Optional[float],
                          max_flow: Optional[float]):
        # add the new commodities to the set
        self._commodities.update(node.commodities_in)
        self._commodities.update(node.commodities_out)
        # check that the node has a unique identifier and append it to the designed internal data structure
        for kind, nodes in self._nodes.items():
            assert node not in nodes, f"There is already a {kind} node '{node.name}', please use another identifier"
        node_set = self._nodes.get(node.kind, set())
        node_set.add(node)
        self._nodes[node.kind] = node_set
        # if the node is not a source (supplier), retrieve the node parent and check that is exists
        if parents is None:
            return
        parents = [parents] if isinstance(parents, str) else parents
        assert len(parents) > 0, f"{node.kind.title()} node must have at least one parent"
        for name in parents:
            parent = self.nodes().get(name)
            assert parent is not None, f"Parent node '{name}' has not been added yet"
            # create an edge instance using the parent as source and the new node as destination
            edge = Edge(
                _plant=self,
                _source=parent,
                _destination=node,
                commodity=commodity,
                min_flow=min_flow,
                max_flow=max_flow
            )
            self._edges.add(edge)

    def add_extremity(self,
                      kind: Literal['customer', 'purchaser', 'supplier'],
                      name: str,
                      commodity: str,
                      predictions: Union[float, Iterable[float]],
                      variance: Callable[[np.random.Generator, pd.Series], float] = lambda rng, series: 0.0,
                      parents: Union[None, str, Iterable[str]] = None) -> ExtremityNode:
        """Adds an extremity node (supplier, client, purchaser) to the plant topology.

        :param kind:
            The kind of extremity node, either 'customer', 'purchaser', or 'supplier'.

        :param name:
            The name of the extremity node.

        :param commodity:
            The identifier of the commodity that it supplies, which must have been registered before.

        :param predictions:
            The vector of predicted selling prices for the commodity during the time horizon, or a float if the
            predictions are constant throughout the simulation. If an iterable is passed, it must have the same length
            of the time horizon.

        :param variance:
            A function f(<rng>, <series>) -> <variance> which defines the variance model of the true prices
            respectively to the predictions.
            The random number generator is used to get reproducible results, while the input series represents the
            vector of previous values indexed by the datetime information passed to the plant; the last value of the
            series is always a nan value, and it will be computed at the successive iteration based on the respective
            predicted price and the output of this function.
            Indeed, the function must return a real number <eps> which represents the delta between the predicted and
            the true price; for an input series with length L, the true price will be eventually computed as:
                true = self.prices[L] + <eps>

        :param parents:
            The identifier of the parent nodes that are connected with the input of this extremity node, or None in
            case kind == 'supplier'.

        :return:
            The added extremity node.
        """
        if isinstance(predictions, float):
            predictions = np.ones_like(self._horizon) * predictions
        else:
            predictions = np.array(predictions)
        # create an internal supplier node and add it to the internal data structure and the graph
        if kind == 'supplier':
            assert parents is None, f"Supplier node {name} cannot accept parents"
            node = Supplier(
                _plant=self,
                name=name,
                commodity=commodity,
                _predictions=predictions,
                _variance_fn=variance
            )
        elif kind == 'purchaser':
            assert parents is not None, f"Purchaser node {name} must have parents"
            node = Purchaser(
                _plant=self,
                name=name,
                commodity=commodity,
                _predictions=predictions,
                _variance_fn=variance
            )
        elif kind == 'customer':
            assert parents is not None, f"Customer node {name} must have parents"
            node = Customer(
                _plant=self,
                name=name,
                commodity=commodity,
                _predictions=predictions,
                _variance_fn=variance
            )
        else:
            raise AssertionError(f"Unknown extremity node kind {kind}")
        self._check_and_update(
            node=node,
            parents=parents,
            commodity=commodity,
            min_flow=0.0,
            max_flow=float('inf')
        )
        return node

    def add_machine(self,
                    name: str,
                    parents: Union[str, Iterable[str]],
                    setpoint: Union[Setpoint, pd.DataFrame],
                    discrete_setpoint: bool = False,
                    max_starting: Optional[Tuple[int, int]] = None,
                    cost: float = 0.0,
                    min_flow: float = 0.0,
                    max_flow: float = float('inf')) -> Machine:
        """Adds a machine node to the topology.

        :param name:
            The name of the machine node.

        :param parents:
            The identifier of the parent nodes that are connected with the input of this machine node.

        :param setpoint:
            Either a dictionary of type {'setpoint': [...], <output_commodity_i>: [...], ...} where 'setpoint'
            represent the data index and <output_commodity_i> is the name of each output commodity generated by the
            machine with the list of respective flows, or a pandas dataframe where the index is a series of floating
            point values that indicate the input commodity flow, while the columns should be named after the output
            commodity and contain the respective output flows.

        :param discrete_setpoint:
            Whether the setpoint is discrete or continuous.

        :param max_starting:
            A tuple <N, T> where N is the maximal number of times that the machine can be switched on in T units.

        :param cost:
            The cost to operate the machine.

        :param min_flow:
            The minimal flow of commodity that can pass in the edge.

        :param max_flow:
            The maximal flow of commodity that can pass in the edge.

        :return:
            The added machine node.
        """
        # convert setpoint to a standard format (pd.Series)
        if isinstance(setpoint, dict):
            assert set(setpoint.keys()) == {'setpoint', 'input', 'output'}, \
                f"Setpoint dictionary expects three keys ('setpoint', 'input', 'output'), got {set(setpoint)}"
            assert len(setpoint['input']) == 1, \
                f"Machines taking multiple input commodities are not supported, got inputs {set(setpoint['input'])}"
            index = pd.Index(setpoint['setpoint'], dtype=float, name='setpoint')
            input_flows = pd.DataFrame({c: f for c, f in setpoint['input'].items()}, dtype=float, index=index)
            output_flows = pd.DataFrame({c: f for c, f in setpoint['output'].items()}, dtype=float, index=index)
            setpoint = pd.concat((input_flows, output_flows), keys=['input', 'output'], axis=1)
        # create an internal machine node and add it to the internal data structure and the graph
        machine = Machine(
            _plant=self,
            name=name,
            _setpoint=setpoint,
            discrete_setpoint=discrete_setpoint,
            max_starting=max_starting,
            cost=cost
        )
        self._check_and_update(
            node=machine,
            parents=parents,
            commodity=setpoint['input'].columns[0],
            min_flow=min_flow,
            max_flow=max_flow
        )
        return machine

    def add_storage(self,
                    name: str,
                    commodity: str,
                    capacity: float,
                    parents: Union[str, List[str]],
                    dissipation: float = 0.0,
                    rates: Union[None, float, Tuple[float, float]] = None,
                    min_flow: float = 0.0,
                    max_flow: float = float('inf')) -> Storage:
        """Adds a storage node to the topology.

        :param name:
            The name of the storage node.

        :param commodity:
            The commodity stored by the storage node.

        :param capacity:
            The storage capacity, which must be a strictly positive and finite number.

        :param parents:
            The identifier of the parent nodes that are connected with the input of this storage node.

        :param dissipation:
            The dissipation rate of the storage at every time unit, which must be a float in [0, 1].

        :param rates:
            Either a tuple (charge_rate, discharge_rate) containing the maximal charge rate (input flow) and discharge
            rate (output flow) in a time unit, or a single float value in case charge_rate == discharge_rate. If None
            is passed, the charge and discharge rates are set to capacity.

        :param min_flow:
            The minimal flow of commodity that can pass in the edge.

        :param max_flow:
            The maximal flow of commodity that can pass in the edge.

        :return:
            The added storage node.
        """
        # handle rates
        if rates is None:
            charge_rate, discharge_rate = capacity, capacity
        elif isinstance(rates, tuple):
            charge_rate, discharge_rate = rates, rates
        else:
            charge_rate, discharge_rate = rates
        # create an internal machine node and add it to the internal data structure and the graph
        storage = Storage(
            _plant=self,
            name=name,
            commodity=commodity,
            capacity=capacity,
            dissipation=dissipation,
            charge_rate=charge_rate,
            discharge_rate=discharge_rate
        )
        self._check_and_update(
            node=storage,
            parents=parents,
            commodity=commodity,
            min_flow=min_flow,
            max_flow=max_flow
        )
        return storage

    def draw(self,
             figsize: Tuple[int, int] = (16, 9),
             node_pos: Union[str, List[Iterable[Optional[str]]], Dict[str, Any]] = 'lp',
             node_colors: Union[None, str, Dict[str, str]] = None,
             node_markers: Union[None, str, Dict[str, str]] = None,
             edge_colors: Union[str, Dict[str, str]] = 'black',
             edge_styles: Union[str, Dict[str, str]] = 'solid',
             node_size: float = 30,
             edge_width: float = 2,
             legend: bool = True):
        """Draws the plant topology.

        :param figsize:
            The matplotlib figsize parameter.

        :param node_pos:
            If the string 'sp' is passed, arranges the nodes into layers using breadth first search to get the shortest
            path. If the string 'lp' is passed, arranges the nodes into layers using negative unitary cost to get the
            longest  path. If a list is passed, it is considered as a mapping <layer: nodes> representing in which
            layer to display the nodes (None values can be used to add placeholder nodes). If a dictionary is passed,
            it is considered as a mapping <node: position> representing where exactly to display each node.

        :param node_colors:
            Either a string representing the color of the nodes, or a dictionary {kind: color} which associates a color
            to each node kind ('supplier', 'client', 'machine').

        :param node_markers:
            Either a string representing the shape of the nodes, or a dictionary {kind: shape} which associates a shape
            to each node kind ('supplier', 'client', 'machine', 'storage').

        :param edge_colors:
            Either a string representing the color of the edges, or a dictionary {commodity: color} which associates a
            color to each commodity that flows in an edge.

        :param edge_styles:
            Either a string representing the style of the edges, or a dictionary {commodity: style} which associates a
            style to each commodity that flows in an edge.

        :param node_size:
            The size of the nodes.

        :param edge_width:
            The width of the edges and of the node's borders.

        :param legend:
            Whether to plot a legend or not.
        """
        # retrieve plant info, build the figure, and compute node positions
        nodes = self.nodes(indexed=True)
        graph = self.graph(attributes=False)
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, tight_layout=True)
        pos = drawing.get_node_positions(graph=graph, sources=nodes['supplier'].keys(), node_pos=node_pos)
        # retrieve nodes' styling information and draw them accordingly
        labels = []
        styles = drawing.get_node_style(colors=node_colors, markers=node_markers)
        for kind, node_list in nodes.items():
            drawing.draw_nodes(
                graph=graph,
                pos=pos,
                nodes=node_list.keys(),
                style=styles[kind],
                size=node_size,
                width=edge_width,
                ax=ax
            )
            handler = drawing.build_node_label(kind=kind, style=styles[kind])
            labels.append(handler)
        # retrieve edges' styling information and draw them accordingly
        styles = drawing.get_edge_style(colors=edge_colors, shapes=edge_styles, commodities=list(self._commodities))
        edges = pd.DataFrame([{'commodity': e.commodity, 'edge': e} for e in self.edges().values()])
        for commodity, edge_list in edges.groupby('commodity'):
            commodity = str(commodity)
            drawing.draw_edges(
                graph=graph,
                pos=pos,
                edges={(e.source, e.destination) for e in edge_list['edge']},
                style=styles[commodity],
                size=node_size,
                width=edge_width,
                ax=ax
            )
            handler = drawing.build_edge_label(commodity=commodity, style=styles[commodity])
            labels.append(handler)
        # plot the legend if necessary, and eventually show the result
        if legend:
            plt.legend(handles=labels, prop={'size': 20})
        plt.show()

    def run(self,
            plan: Union[Plan, pd.DataFrame],
            action: Union[str, RecourseAction, Callable[[Any], Plan]]) -> execution.SimulationOutput:
        """Runs a simulation up to the time horizon using the given plan.

        :param plan:
            The energetic plan of the power plant defined as vectors of edge flows and machine states. It can be a
            dictionary <machine | edge: states | flows> that maps each machine (indexed by its name) to either a fixed
            state or an iterable of such which should match the time horizon of the simulation, and each edge (indexed
            by the tuple of source and destination) to either a fixed flow or an iterable of such which should match
            the time horizon of the simulation. Otherwise, it can be a dataframe where each column is indexed by the
            machine name of the edge key, and the value is a vector of states or flows, respectively, with the index
            of the dataframe that matches the time horizon of the simulation.

        :param action:
            If a RecourseAction object is passed, it is used to build the real plan at each step.
            If a function f(plant) -> plan, this is wrapped into a CallableRecourseAction object.
            If a string is passed, this is interpreted as the solver to use in the default greedy recourse action.

        :return:
            A SimulationOutput object containing all the information about true prices, demands, setpoints, and storage.
        """
        assert self._step == -1, "Simulation for this plant was already run, create a new instance to run another one"
        execution.check_plant(plant=self)
        # handle recourse action (if None use default action, if Callable build a custom recourse action)
        if isinstance(action, str):
            action = DefaultRecourseAction(solver=action)
        elif isinstance(action, Callable):
            action = CallableRecourseAction(action=action)
        action.build(plant=self)
        # retrieve data and process input plan
        nodes = self.nodes()
        edges = self.edges()
        machines = self.machines
        datatypes = {**edges, **nodes}
        plan = execution.process_plan(plan=plan, machines=machines, edges=edges, horizon=self._horizon)
        # start the simulation
        for _, row in plan.iterrows():
            self._step += 1
            # update the simulation objects before the recourse action
            for datatype in datatypes.values():
                datatype.update(rng=self._rng, flows=row['flows'], states=row['states'])
            # compute the updated flows and states using the recourse action
            updated_states, updated_flows = check_plan(
                plan=action.execute(),
                machines=machines,
                edges=edges
            )
            # update the simulation objects after the recourse action
            for datatype in datatypes.values():
                datatype.step(flows=updated_flows, states=updated_states)
        return execution.build_output(nodes=nodes.values(), edges=edges.values(), horizon=self._horizon)
