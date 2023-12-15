import copy
from typing import Optional, Dict, Tuple, Callable, Iterable, Set, List, Any, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ppsim import utils
from ppsim.datatypes import Node, Client, Machine, Supplier, Edge, Storage, Purchaser, Customer
from ppsim.plant import drawing, execution
from ppsim.utils.typing import Plan, Flows, Setpoint


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
        return {s.name: s for s in self._nodes.get(Supplier.kind, set())}

    @property
    def clients(self) -> Dict[str, Client]:
        """The client nodes in the plant."""
        return {c.name: c for c in self._nodes.get(Client.kind, set())}

    @property
    def machines(self) -> Dict[str, Machine]:
        """The machine nodes in the plant."""
        return {m.name: m for m in self._nodes.get(Machine.kind, set())}

    @property
    def storages(self) -> Dict[str, Storage]:
        """The storage nodes in the plant."""
        return {s.name: s for s in self._nodes.get(Storage.kind, set())}

    def nodes(self, indexed: bool = False) -> Dict[str, Union[Node, Dict[str, Node]]]:
        """Returns the nodes in the plant.

        :param indexed:
            Whether or not to index the dictionary by type.

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
            Either a dictionary <(source, destination), Dict[commodity, edge]> with nodes pairs as key, or a dictionary
            <(source, destination, commodity), edge> with a triplet of nodes and commodity as key.
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
            Whether or not to include node/edge attributes.

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

    def copy(self):
        """Copies the plant object.

        :return:
            A copy of the plant object.
        """
        return copy.deepcopy(self)

    def _check_and_update(self,
                          node: Node,
                          parents: Union[None, str, Iterable[str]],
                          min_flow: Optional[float],
                          max_flow: Optional[float],
                          integer: Optional[bool]):
        # add the new commodities to the set
        if node.commodity_in is not None:
            self._commodities.add(node.commodity_in)
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
            assert node.commodity_in in parent.commodities_out, \
                f"Parent node '{parent.name}' should return commodity '{node.commodity_in}', " \
                f"but it returns {parent.commodities_out}"
            # create an edge instance using the parent as source and the new node as destination
            edge = Edge(
                _plant=self,
                _source=parent,
                _destination=node,
                min_flow=min_flow,
                max_flow=max_flow,
                integer=integer
            )
            self._edges.add(edge)
            # append parent and children to the respective lists
            parent.append(edge)
            node.append(edge)

    def add_supplier(self,
                     name: str,
                     commodity: str,
                     predictions: Union[float, Iterable[float]],
                     variance: Callable[[np.random.Generator, pd.Series], float] = lambda rng, series: 0.0) -> Supplier:
        """Adds a supplier node to the plant topology.

        :param name:
            The name of the supplier node.

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

        :return:
            The added supplier node.
        """
        predictions = np.ones_like(self._horizon) * predictions if isinstance(predictions, float) else predictions
        # create an internal supplier node and add it to the internal data structure and the graph
        supplier = Supplier(
            _plant=self,
            name=name,
            commodity=commodity,
            _predictions=predictions,
            _variance_fn=variance
        )
        self._check_and_update(node=supplier, parents=None, min_flow=None, max_flow=None, integer=None)
        return supplier

    def add_client(self,
                   name: str,
                   commodity: str,
                   parents: Union[str, Iterable[str]],
                   predictions: Union[float, Iterable[float]],
                   purchaser: bool = False,
                   variance: Callable[[np.random.Generator, pd.Series], float] = lambda rng, series: 0.0) -> Client:
        """Adds a client node to the plant topology.

        :param name:
            The name of the client node.

        :param commodity:
            The identifier of the commodity that it requests, which must have been registered before.

        :param parents:
            The identifier of the parent nodes that are connected with the input of this client node.

        :param predictions:
            Either the vector of predicted buying prices (in case the client node is a purchaser) or the vector of
            predicted demands (in case the client is a customer). If a float is passed, the predictions are constant
            throughout the simulation. If an iterable is passed, it must have the same length of the time horizon.

        :param purchaser:
            Whether the client node buys the commodity at a given price (series = prices) or requires that an exact
            amount of commodities is sent to it according to its demands (series = demands).

        :param variance:
            A function f(<rng>, <series>) -> <variance> which defines the variance model of the true pries/demands
            respectively to the predictions.
            The random number generator is used to get reproducible results, while the input series represents the
            vector of previous values indexed by the datetime information passed to the plant; the last value of the
            series is always a nan value, and it will be computed at the successive iteration based on the respective
            predicted demand and the output of this function.
            Indeed, the function must return a real number <eps> which represents the delta between the predicted and
            the true demand; for an input series with length L, the true demand will be eventually computed as:
                true = self.demands[L] + <eps>

        :return:
            The added client node.
        """
        predictions = np.ones_like(self._horizon) * predictions if isinstance(predictions, float) else predictions
        # create an internal client node (with specified type) and add it to the internal data structure and the graph
        if purchaser:
            client = Purchaser(
                _plant=self,
                name=name,
                commodity=commodity,
                _predictions=predictions,
                _variance_fn=variance
            )
        else:
            client = Customer(
                _plant=self,
                name=name,
                commodity=commodity,
                _predictions=predictions,
                _variance_fn=variance
            )
        self._check_and_update(node=client, parents=parents, min_flow=0.0, max_flow=float('inf'), integer=False)
        return client

    def add_machine(self,
                    name: str,
                    commodity: str,
                    parents: Union[str, Iterable[str]],
                    setpoint: Union[Setpoint, pd.DataFrame],
                    discrete_setpoint: bool = False,
                    max_starting: Optional[Tuple[int, int]] = None,
                    cost: float = 0.0,
                    min_flow: float = 0.0,
                    max_flow: float = float('inf'),
                    integer: bool = False) -> Machine:
        """Adds a machine node to the topology.

        :param name:
            The name of the machine node.

        :param commodity:
            The input commodity of the machine.

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

        :param integer:
            Whether the flow must be integer or not.

        :return:
            The added machine node.
        """
        # convert setpoint to a standard format (pd.Series)
        if isinstance(setpoint, dict):
            setpoint = pd.DataFrame(setpoint).set_index('setpoint')
        # create an internal machine node and add it to the internal data structure and the graph
        machine = Machine(
            _plant=self,
            name=name,
            _setpoint=setpoint,
            commodity=commodity,
            discrete_setpoint=discrete_setpoint,
            max_starting=max_starting,
            cost=cost
        )
        self._check_and_update(node=machine, parents=parents, min_flow=min_flow, max_flow=max_flow, integer=integer)
        return machine

    def add_storage(self,
                    name: str,
                    commodity: str,
                    parents: Union[str, List[str]],
                    capacity: float = float('inf'),
                    dissipation: float = 0.0,
                    rates: Union[float, Tuple[float, float]] = float('inf'),
                    min_flow: float = 0.0,
                    max_flow: float = float('inf'),
                    integer: bool = False) -> Storage:
        """Adds a storage node to the topology.

        :param name:
            The name of the storage node.

        :param commodity:
            The commodity stored by the storage node.

        :param parents:
            The identifier of the parent nodes that are connected with the input of this storage node.

        :param capacity:
            The storage capacity, which must be a strictly positive number.

        :param dissipation:
            The dissipation rate of the storage at every time unit, which must be a float in [0, 1].

        :param rates:
            Either a tuple (charge_rate, discharge_rate) containing the maximal charge rate (input flow) and discharge
            rate (output flow) in a time unit, or a single float value in case charge_rate == discharge_rate.

        :param min_flow:
            The minimal flow of commodity that can pass in the edge.

        :param max_flow:
            The maximal flow of commodity that can pass in the edge.

        :param integer:
            Whether the flow must be integer or not.

        :return:
            The added storage node.
        """
        charge_rate, discharge_rate = rates if isinstance(rates, tuple) else (rates, rates)
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
        self._check_and_update(node=storage, parents=parents, min_flow=min_flow, max_flow=max_flow, integer=integer)
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
            action_fn: Optional[Callable[[Any, nx.DiGraph], Flows]] = None) -> execution.SimulationOutput:
        """Runs a simulation up to the time horizon using the given plan.

        :param plan:
            The energetic plan of the power plant defined as vectors of flows within the edges of the power plant.
            It can be a dictionary <(source, destination), flows> that maps each edge (source, destination) to either
            a fixed flow or an iterable of such which should match the time horizon of the simulation, or it can be a
            dataframe where the index matches the time horizon of the simulation and the columns are indexed by tuple
            (source, destination).

        :param action_fn:
            A function f(step: int, graph: nx.DiGraph) -> updated_flows: Dict[Tuple[str, str], float].
            This function models the recourse action for a given step (step), which is provided as input together with
            the topology of the power plant as returned by plant.graph(attributes=True), with an additional attribute
            'flow' stored in the edges which represent the flow provided by the original plan. The function must return
            a dictionary {<edge>: <updated_flow>} where an <edge> is identified by the tuple of the names of the node
            that is connecting, and <updated_flow> is a floating point value of the actual flow.
            If None is passed, a default greedy recourse action is used instead.

        :return:
            A SimulationOutput object containing all the information about true prices, demands, setpoints, and storage.
        """
        nodes, edges = self.nodes(), self.edges()
        action_fn = execution.default_action if action_fn is None else action_fn
        plan = execution.process_plan(plan=plan, edges=edges.keys(), horizon=self._horizon)
        graph = self.graph(attributes=True)
        for step, (_, row) in enumerate(plan.iterrows()):
            # updates the action graph with the last flow
            for (source, destination), flow in row.items():
                graph[source][destination]['flow'] = flow
            flows = action_fn(step, graph)
            execution.run_update(nodes=nodes.values(), edges=edges.values(), flows=flows, rng=self._rng)
        return execution.build_output(nodes=nodes.values(), edges=edges.values(), horizon=self._horizon)
