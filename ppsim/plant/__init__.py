from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Tuple, List, Callable, Iterable, Set, Type

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from ppsim.datatypes import Node, Client, Machine, Supplier, Edge


class Plant:
    """Defines a power plant based on its topology, involved commodities, and predicted prices and demands."""

    @dataclass(frozen=True)
    class _EdgesInfo:
        """Internal data structure to pair each commodity to a dictionary of edges, indexed by edge number."""

        commodity: str = field()
        """The given commodity."""

        edges: Dict[int, List[Edge]] = field(default_factory=dict)
        """A dictionary {i: edges} where <i> means that those are the i-th edges connecting the same nodes.
        This information is useful only to plot progressively increasing curvatures in case of multi-edges."""

    COLORS: Dict[Type, str] = {
        Supplier: '#FFEAB8',
        Machine: '#FF700A',
        Client: '#A6DEAE'
    }
    """Default dictionary of node colors."""

    SHAPES: Dict[Type, str] = {
        Supplier: '>',
        Client: '<',
        Machine: 'o'
    }
    """Default dictionary of node markers shapes."""

    def __init__(self, horizon: int | list | np.ndarray | pd.Series | pd.Index):
        """
        :param horizon:
            The time horizon of the simulation.
            If an integer is passed, the index will be {0, ..., horizon - 1}.
            Otherwise, an explicit list, numpy array or pandas series can be passed.
        """
        # convert horizon to a standard format (pd.Index)
        if not isinstance(horizon, Iterable):
            assert horizon > 0, f"The time horizon must be a positive integer, got {horizon}"
            horizon = np.arange(horizon)
        horizon = pd.Index(horizon)

        self.horizon: pd.Index = horizon
        """A pandas series representing the time index of the simulation."""

        self.suppliers: Set[Supplier] = set()
        """The supplier nodes in the plant."""

        self.clients: Set[Client] = set()
        """The client nodes in the plant."""

        self.machines: Set[Machine] = set()
        """The machine nodes in the plant."""

        self._graph: nx.DiGraph = nx.MultiDiGraph()
        """An internal graph object representing the topology of the plant."""

        self._commodities: Dict[str, Plant._EdgesInfo] = dict()
        """The plant edges (and additional info) indexed by commodity identifier."""

    def nodes(self, indexed: bool = False) -> Dict[str, Set[Node]] | Set[Node]:
        """Returns the nodes in the plant.

        :param indexed:
            Whether to return a dictionary of nodes indexed by type (supplier, client, machine) or a flattened list.

        :return:
            Either a dictionary <type, node> or a list of nodes.
        """
        return {
            Supplier.kind: self.suppliers,
            Client.kind: self.clients,
            Machine.kind: self.machines
        } if indexed else {
            *self.suppliers,
            *self.machines,
            *self.clients
        }

    def edges(self,
              sources: None | str | Iterable[str] = None,
              destinations: None | str | Iterable[str] = None,
              commodities: None | str | Iterable[str] = None,
              triplets: bool = True) -> Dict[Edge.EdgeID | Tuple[str, str], Edge | Dict[str, Edge]]:
        """Returns the edges in the plant indexed either via nodes, or via nodes and commodity.

        :param sources:
            The identifier of the source nodes, used to filter just the edges starting from this node.

        :param destinations:
            The identifier of the destination nodes, used to filter just the edges ending in this node.

        :param commodities:
            The identifier of the commodities, used to filter just the edges ending in this node.

        :param triplets:
            Whether to index the output dictionary via the pair (source, destination), in which case a single Edge
            object is paired to the key, or via the triplet (source, destination, commodity) in which case a dictionary
            of <commodity, edge> mappings is paired with the key.

        :return:
            Either a dictionary <(source, destination), Dict[commodity, edge]> with nodes pairs as key, or a dictionary
            <(source, destination, commodity), edge> with a triplet of nodes and commodity as key.
        """
        edges = {}
        edge_list = self._graph.edges(sources, data='attr')
        # process destination check
        match destinations:
            case None:
                check_dest = lambda d: True
            case str():
                check_dest = lambda d: d == destinations
            case _:
                destinations = set(destinations)
                check_dest = lambda d: d in destinations
        # process commodities check
        match commodities:
            case None:
                check_edge = lambda e: True
            case str():
                check_edge = lambda e: e.commodity == commodities
            case _:
                commodities = set(commodities)
                check_edge = lambda e: e.commodity in commodities
        # build data structure with the correct index
        if triplets:
            for sour, dest, edge in edge_list:
                if check_dest(dest) and check_edge(edge):
                    edges[(sour, dest, edge.commodity)] = edge
        else:
            for sour, dest, edge in edge_list:
                if check_dest(dest) and check_edge(edge):
                    val = edges.get((sour, dest), dict())
                    edges[(sour, dest)] = {**val, edge.commodity: edge}
        return edges

    def _check_and_append_node(self, node: Node, structure: Set[Node]):
        assert node not in self.suppliers, f"There is already a supplier '{node.key()}', please use another identifier"
        assert node not in self.clients, f"There is already a client '{node.key()}', please use another identifier"
        assert node not in self.machines, f"There is already a machine '{node.key()}', please use another identifier"
        structure.add(node)

    def _add_extremity_node(self,
                            name: str,
                            client: bool,
                            commodity: str,
                            predictions: float | list | np.ndarray | pd.Series,
                            variance: Callable[[np.random.Generator, pd.Series], float] = lambda rng, series: 0.0):
        """Adds a either a client or a supplier node to the plant topology.

        :param name:
            The name of the node.

        :param client:
            Whether the node is a client (True) or a supplier (False).

        :param commodity:
            The identifier of the commodity that it supplies/receives.

        :param predictions:
            The vector of (predicted) prices/demands for the commodity during the time horizon, or a float if the
            predictions are constant throughout the simulation. If an iterable is passed, it must have the same length
            of the time horizon.

        :param variance:
            A function f(<rng>, <series>) -> <variance> which defines the variance model of the true values
            respectively to the predictions.
            The random number generator is used to get reproducible results, while the input series represents the
            vector of previous values indexed by the datetime information passed to the plant; the last value of the
            series is always a nan value, and it will be computed at the successive iteration based on the respective
            predicted value and the output of this function.
            Indeed, the function must return a real number <eps> which represents the delta between the predicted and
            the true values; for an input series with length L, the true value will be eventually computed as:
                true = self.predictions[L] + <eps>
        """
        # convert predictions to a standard format (pd.Series)
        predictions = pd.Series(predictions, dtype=float, index=self.horizon)
        # check if the commodity exists or create a new entry in the internal data structure otherwise
        if commodity not in self._commodities:
            self._commodities[commodity] = Plant._EdgesInfo(commodity=commodity)
        # create either a client or a supplier node instance and add it to the internal data structure and the graph
        if client:
            node = Client(name=name, commodity_in=commodity, demands=predictions, _variance=variance)
            self._check_and_append_node(node=node, structure=self.clients)
        else:
            node = Supplier(name=name, commodity_out=commodity, prices=predictions, _variance=variance)
            self._check_and_append_node(node=node, structure=self.suppliers)
        self._graph.add_node(name, attr=node)

    def add_supplier(self,
                     name: str,
                     commodity: str,
                     prices: float | list | np.ndarray | pd.Series,
                     variance: Callable[[np.random.Generator, pd.Series], float] = lambda rng, series: 0.0):
        """Adds a supplier node to the plant topology.

        :param name:
            The name of the supplier node.

        :param commodity:
            The identifier of the commodity that it supplies, which must have been registered before.

        :param prices:
            The vector of (predicted) prices for the commodity during the time horizon, or a float if the predictions
            are constant throughout the simulation. If an iterable is passed, it must have the same length of the time
            horizon.

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
        """
        self._add_extremity_node(name=name, commodity=commodity, predictions=prices, variance=variance, client=False)

    def add_client(self,
                   name: str,
                   commodity: str,
                   demands: float | list | np.ndarray | pd.Series,
                   variance: Callable[[np.random.Generator, pd.Series], float] = lambda rng, series: 0.0):
        """Adds a client node to the plant topology.

        :param name:
            The name of the client node.

        :param commodity:
            The identifier of the commodity that it requests, which must have been registered before.

        :param demands:
            The vector of (predicted) demands for the commodity during the time horizon, or a float if the demands are
            constant throughout the simulation. If an iterable is passed, it must have the same length of the time
            horizon.

        :param variance:
            A function f(<rng>, <series>) -> <variance> which defines the variance model of the true demands
            respectively to the predictions.
            The random number generator is used to get reproducible results, while the input series represents the
            vector of previous values indexed by the datetime information passed to the plant; the last value of the
            series is always a nan value, and it will be computed at the successive iteration based on the respective
            predicted demand and the output of this function.
            Indeed, the function must return a real number <eps> which represents the delta between the predicted and
            the true demand; for an input series with length L, the true demand will be eventually computed as:
                true = self.demands[L] + <eps>
        """
        self._add_extremity_node(name=name, commodity=commodity, predictions=demands, variance=variance, client=True)

    def add_machine(self,
                    name: str,
                    setpoint: None | set | list | Tuple[float, float] | Callable[[Any], bool] = None,
                    cost: float | Callable[[Any], float] = 0.0,
                    capacity: float | Dict[str, float] | Callable[[Any, str], float] = 0.0,
                    operation: Callable[[Any, Dict[str, float]], Dict[str, float]] = lambda state, commodities: {}):
        """Adds a machine node to the topology.

        :param name:
            The name of the machine node.

        :param setpoint:
            Defines the valid setpoint of the machine. It can be either a set/list of admitted values, a tuple <lb, ub>
            of lower and upper bounds, or a custom function f(state) -> bool which explicitly checks whether a state is
            part of the setpoint (i.e., a valid state) or not. If None is passed, every possible value is admitted.

        :param cost:
            Either a fixed cost for the machine to be operated, or a function f(state) -> cost which returns the cost
            of operating the machine for a certain state state.

        :param capacity:
            Either a fixed capacity of the machine which is the same for all the output commodities, a fixed dictionary
            <commodity, capacity> which defines the capacity of the machine for each output commodity, or a custom
            function f(state, commodity) -> capacity which returns the maximal storing capacity of each commodity.

        :param operation:
            A function f(state, commodities_in) -> commodities_out representing the conversion operation for this state.
        """
        # process the setpoint
        match setpoint:
            case None:
                setpoint_fn = lambda state: True
            case set():
                setpoint_fn = lambda state: state in setpoint
            case list():
                setpoint = set(setpoint)
                setpoint_fn = lambda state: state in setpoint
            case tuple():
                assert len(setpoint) == 2, "If a tuple is passed, it must have two elements, i.e., <lb> and <ub>"
                setpoint_fn = lambda state: setpoint[0] <= state <= setpoint[1]
            case _:
                setpoint_fn = setpoint
        # process the capacities
        match capacity:
            case float():
                capacity_fn = lambda state, commodity: capacity
            case dict():
                capacity_fn = lambda state, commodity: capacity[commodity]
            case _:
                capacity_fn = capacity
        # process the cost
        match cost:
            case float():
                cost_fn = lambda state: cost
            case _:
                cost_fn = cost
        # creates a node instance and add it both to the internal data structure and to the graph
        node = Machine(name=name, _setpoint=setpoint_fn, _cost=cost_fn, _capacity=capacity_fn, _operation=operation)
        self._check_and_append_node(node=node, structure=self.machines)
        self._graph.add_node(name, attr=node)

    def add_edge(self,
                 source: str,
                 destination: str,
                 commodity: Optional = None,
                 min_flow: float = 0.0,
                 max_flow: float = float('inf'),
                 integer: bool = False):
        """Adds an edge between two nodes in the topology.

        :param source:
            The source node, which must be either a supplier or a machine node.

        :param destination:
            The destination node, which must be either a client or a machine node.

        :param commodity:
            The type of commodity that flows in the node, or None if the commodity type can be inferred from the nodes.
            If both the source and the destination nodes are machines, the commodity must be specified.

        :param min_flow:
            The minimal flow of commodity that can pass in the edge.

        :param max_flow:
            The maximal flow of commodity that can pass in the edge.

        :param integer:
            Whether the flow must be integer or not.
        """
        # retrieve the source and destination nodes and check that they exist
        sn, dn = self._graph.nodes.get(source), self._graph.nodes.get(destination)
        assert sn is not None, f"Source node '{source}' is not in the plant, please insert it first"
        assert dn is not None, f"Destination node '{destination}' is not in the plant, please insert it first"
        # retrieve the output commodity of the source node and the input commodity of the destination node
        sn, dn = sn['attr'], dn['attr']
        sc, dc = sn.commodity_out, dn.commodity_in
        # perform consistency checks to guarantee that only correct edges are included
        match sc, dc, commodity:
            case False, _, _:
                raise AssertionError(f"The source must supply a commodity (supplier or machine node)")
            case _, False, _:
                raise AssertionError("The destination must receive a commodity (client or machine node)")
            case True, True, None:
                raise AssertionError("Both nodes can accept any commodity, please specify its identifier.")
            case True, True, str():
                pass
            case str(), True, None:
                commodity = sc
            case str(), True, str():
                assert sc == commodity, "The output commodity of the source must match the given one"
            case True, str(), None:
                commodity = dc
            case True, str(), str():
                assert dc == commodity, "The input commodity of the destination must match the given one"
            case str(), str(), None:
                assert sc == dc, "The output commodity of the source must match the input of the destination"
                commodity = sc
            case str(), str(), str():
                assert sc == dc == commodity, "The input/output commodities of the nodes must match the given one"
        # retrieve the commodity and edges information if the commodity exists, otherwise create a new entry
        info = self._commodities.get(commodity)
        if info is None:
            info = Plant._EdgesInfo(commodity=commodity)
            self._commodities[commodity] = info
        # for machine nodes, append the commodity identifier in the output/input lists
        if isinstance(sn, Machine):
            sn.append(commodity, out=True)
        if isinstance(dn, Machine):
            dn.append(commodity, out=False)
        # retrieve the edges involving the same nodes to check that none of them is linked to the same commodity
        edges = self._graph.get_edge_data(source, destination, default=dict())
        for e in edges.values():
            assert e['attr'].commodity != commodity, f"There is already and edge ({source}, {destination}, {commodity})"
        # create an edge instance and include it in the graph
        e = Edge(nodes=(sn, dn), commodity=info.commodity, min_flow=min_flow, max_flow=max_flow, integer=integer)
        self._graph.add_edge(source, destination, attr=e)
        # use the number of edges involving the same nodes to index the edge in the internal data structure,
        # this information will be useful to plot progressively increasing curvatures in case of multi-edges
        n = len(edges) + 1
        if n in info.edges:
            info.edges[n].append(e)
        else:
            info.edges[n] = [e]

    def draw(self,
             figsize: Tuple[int, int] = (16, 9),
             node_colors: str | Dict[str, str] = 'default',
             node_shapes: str | Dict[str, str] = 'default',
             edge_colors: str | Dict[str, str] = 'black',
             edge_styles: str | Dict[str, str] = 'solid',
             node_size: float = 30,
             edge_width: float = 2,
             legend: bool = True):
        """Draws the plant topology.

        :param figsize:
            The matplotlib figsize parameter.

        :param node_colors:
            Either a string representing the color of the nodes, or a dictionary {kind: color} which associates a color
            to each node kind ('supplier', 'client', 'machine').

        :param node_shapes:
            Either a string representing the shape of the nodes, or a dictionary {kind: shape} which associates a shape
            to each node kind ('supplier', 'client', 'machine').

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
        # build a list of labels to be included in the legend and prepare the figure
        labels = []
        plt.figure(figsize=figsize)
        # create a copy of the graph in order to include custom attributes
        g = self._graph.copy()
        # traverse the graph from the sources using breadth-first search and label each node with a progressive number
        # the position of the nodes is eventually obtained by layering them respectively to the computed number so that
        # suppliers will be on the left and clients on the right
        sources = [node.name for node in self.suppliers]
        for it, nodes in enumerate(nx.bfs_layers(g, sources=sources)):
            for node in nodes:
                g.nodes[node]['layer'] = it
        pos = nx.multipartite_layout(g, subset_key='layer')
        # build a dictionary of color and shape mappings indexed by node kind
        # in case either the colors or the shapes are not default, include the custom information in the dictionary
        styles = {n: {'color': self.COLORS[n], 'shape': self.SHAPES[n]} for n in self.COLORS.keys()}
        if node_colors != 'default':
            node_colors = {n: node_colors for n in self.COLORS.keys()} if isinstance(node_colors, str) else node_colors
            for n, c in node_colors.items():
                styles[n]['color'] = c
        if node_shapes != 'default':
            node_shapes = {n: node_shapes for n in self.SHAPES.keys()} if isinstance(node_shapes, str) else node_shapes
            for n, s in node_shapes.items():
                styles[n]['shape'] = s
        # draw the nodes with respective colors and shapes according to their node kind in the styles dictionary
        # additionally, append a Line2D object to the labels list that will be used in the legend
        for kind, style in styles.items():
            # create a handle with the correct style, color, and text label
            labels.append(Line2D(
                [],
                [],
                marker=style['shape'],
                markerfacecolor=style['color'],
                markeredgecolor='black',
                linestyle='None',
                markersize=25,
                label=kind.value.title()
            ))
            # draw the subset of nodes with the given kind using the correct style and color
            nodes = self.nodes(indexed=True)
            nx.draw(
                g,
                pos=pos,
                edgelist=[],
                nodelist=[node.name for node in nodes[kind]],
                node_color=style['color'],
                node_shape=style['shape'],
                node_size=node_size * 100,
                linewidths=edge_width,
                with_labels=True,
                edgecolors='k',
                arrows=True
            )
        # build a dictionary of color and style mappings indexed by commodity
        # in this case, it is sufficient to create two different dictionaries and the zip them together
        col = {c: edge_colors for c in self._commodities} if isinstance(edge_colors, str) else edge_colors
        shp = {c: edge_styles for c in self._commodities} if isinstance(edge_styles, str) else edge_styles
        styles = {c: {'color': col.get(c, 'black'), 'style': shp.get(c, 'solid')} for c in self._commodities}
        # draw the edges with respective colors and styles according to their node kind in the styles dictionary
        # additionally, append a Line2D object to the labels list that will be used in the legend
        for commodity, style in styles.items():
            # create a handle with the correct style, color, and text label
            labels.append(Line2D(
                [],
                [],
                lw=2,
                color=style['color'],
                linestyle=style['style'],
                label=commodity.title()
            ))
            # retrieve the subset of edges to be drawn and create a new list of straight (curvature 0) edges
            # these edges are those that:
            #   - are in the first level (curvature 1), but
            #   - have no other edge sharing the same source and destination, and
            #   - have no other edge sharing going in the opposite way
            # indeed, these edges will be the only one connecting their nodes, hence they can be straight
            levels = self._commodities[commodity].edges.copy()
            levels[0] = [
                e for e in levels.get(1, [])
                if len(g.get_edge_data(u=e.source.name, v=e.destination.name)) == 1 and e.nodes[::-1] not in g.edges
            ]
            levels[1] = [e for e in levels.get(1, []) if e not in levels[0]]
            # iterate over the lists of edges indexed by curvature to draw such subset using the correct style and color
            for curvature, edges in levels.items():
                nx.draw(
                    g,
                    pos=pos,
                    nodelist=[],
                    edgelist=[(e.source.name, e.destination.name) for e in edges],
                    edge_color=style['color'],
                    style=style['style'],
                    node_size=node_size * 100,
                    arrowsize=edge_width * 10,
                    width=edge_width,
                    connectionstyle=f'arc3, rad = {-curvature * 0.15}',
                    arrows=True
                )
        # plot the legend if necessary, and eventually show the result
        if legend:
            plt.legend(handles=labels, prop={'size': 20})
        plt.show()
