from typing import Optional, Dict, Tuple, Callable, Iterable, Set, List, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ppsim import utils
from ppsim.datatypes import InternalNode, InternalClient, InternalMachine, InternalSupplier, InternalEdge, \
    InternalStorage, Supplier, Client, Machine, Storage, Node
from ppsim.plant import drawing


class Plant:
    """Defines a power plant based on its topology, involved commodities, and predicted prices and demands."""

    def __init__(self, horizon: int | list | np.ndarray | pd.Series | pd.Index):
        """
        :param horizon:
            The time horizon of the simulation.
            If an integer is passed, the index will be {0, ..., horizon - 1}.
            Otherwise, an explicit list, numpy array or pandas series can be passed.
        """
        # convert horizon to a standard format (pd.Index)
        if not isinstance(horizon, Iterable):
            assert horizon > 0, f"The time horizon must be a strictly positive integer, got {horizon}"
            horizon = np.arange(horizon)
        horizon = pd.Index(horizon)

        self._horizon: pd.Index = horizon
        self._commodities: Set[str] = set()
        self._nodes: Dict[str, InternalNode] = dict()
        self._edges: Dict[Tuple[str, str], InternalEdge] = dict()

    @property
    def horizon(self) -> pd.Index:
        """A pandas series representing the time index of the simulation."""
        return self._horizon.copy(deep=True)

    @property
    def commodities(self) -> Set[str]:
        return {c for c in self._commodities}

    @property
    def suppliers(self) -> Dict[str, Supplier]:
        """The supplier nodes in the plant."""
        return {name: node.exposed for name, node in self._nodes.items() if isinstance(node, InternalSupplier)}

    @property
    def clients(self) -> Dict[str, Client]:
        """The client nodes in the plant."""
        return {name: node.exposed for name, node in self._nodes.items() if isinstance(node, InternalClient)}

    @property
    def machines(self) -> Dict[str, Machine]:
        """The machine nodes in the plant."""
        return {name: node.exposed for name, node in self._nodes.items() if isinstance(node, InternalMachine)}

    @property
    def storages(self) -> Dict[str, Storage]:
        """The storage nodes in the plant."""
        return {name: node.exposed for name, node in self._nodes.items() if isinstance(node, InternalStorage)}

    def nodes(self, indexed: bool = False) -> Dict[str, Node | Dict[str, Node]]:
        """Returns the nodes in the plant.

        :param indexed:
            Whether or not to index the dictionary by type.

        :return:
            Either a dictionary {type: {name: node}} or a simple dictionary {name: node}.
        """
        if indexed:
            output = {}
            for name, node in self._nodes.items():
                sub_dict = output.get(node.kind, dict())
                sub_dict[name] = node.exposed
                output[node.kind] = sub_dict
        else:
            output = {name: node.exposed for name, node in self._nodes.items()}
        return output

    def edges(self,
              sources: None | str | Iterable[str] = None,
              destinations: None | str | Iterable[str] = None,
              commodities: None | str | Iterable[str] = None) -> pd.DataFrame:
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
        # retrieve list of edges (filtered by sources) and get filtering functions for destinations and commodities
        check_sour = utils.get_filtering_function(user_input=sources)
        check_dest = utils.get_filtering_function(user_input=destinations)
        check_edge = utils.get_filtering_function(user_input=commodities)
        # build data structure containing all the necessary information
        edges = [
            (s, d, e.commodity, e.exposed)
            for (s, d), e in self._edges.items()
            if check_sour(s) and check_dest(d) and check_edge(e.commodity)
        ]
        return pd.DataFrame(edges, columns=['source', 'destination', 'commodity', 'edge'])

    def graph(self, attributes: bool = False) -> nx.DiGraph:
        """Builds the graph representing the power plant.

        :param attributes:
            Whether or not to include node/edge attributes (under the key 'attr').

        :return:
            A networkx DiGraph object representing the power plant.
        """
        g = nx.DiGraph()
        if attributes:
            for name, node in self._nodes.items():
                g.add_node(name, attr=node.exposed)
            for (source, destination), edge in self._edges.items():
                g.add_edge(source, destination, attr=edge.exposed)
        else:
            g.add_nodes_from(self._nodes.keys())
            g.add_edges_from(self._edges.keys())
        return g

    def copy(self) -> Any:
        """Copies the plant object.

        :return:
            A copy of the plant object.
        """
        copy = Plant(horizon=self.horizon)
        copy._commodities = set(self._commodities)
        copy._nodes = dict(self._nodes)
        copy._edges = dict(self._edges)
        return copy

    def _check_and_update(self,
                          node: InternalNode,
                          parents: None | str | List[str],
                          min_flow: Optional[float],
                          max_flow: Optional[float],
                          integer: Optional[bool]):
        # add the new commodities to the set
        if node.commodity_in is not None:
            self._commodities.add(node.commodity_in)
        self._commodities.update(node.commodities_out)
        # check that the node has a unique identifier
        alias = self._nodes.get(node.name)
        assert alias is None, f"There is already a {alias.kind} node '{node.name}', please use another identifier"
        # append the node to the graph and to the designed internal data structure
        self._nodes[node.name] = node
        # if the node is not a source (supplier), retrieve the node parent and check that is exists
        if parents is None:
            return
        parents = [parents] if isinstance(parents, str) else parents
        assert len(parents) > 0, f"{node.kind.title()} node must have at least one parent"
        for parent in parents:
            parent = self._nodes.get(parent)
            assert parent is not None, f"Parent node {parent} has not been added yet"
            assert node.commodity_in in parent.commodities_out, \
                f"Parent node '{parent.name}' should return commodity '{node.commodity_in}', " \
                f"but it returns {parent.commodities_out}"
            # create an edge instance using the parent as source and the new node as destination
            edge = InternalEdge(source=parent, destination=node, min_flow=min_flow, max_flow=max_flow,
                                integer=integer)
            self._edges[(parent.name, node.name)] = edge
            # append parent and children to the respective lists
            parent.children.append(node)
            node.parents.append(parent)

    def add_supplier(self,
                     name: str,
                     commodity: str,
                     prices: float | list | np.ndarray | pd.Series,
                     variance: Callable[[np.random.Generator, pd.Series], float] = lambda rng, series: 0.0) -> Supplier:
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

        :return:
            The added supplier node.
        """
        # convert prices to a standard format (pd.Series)
        prices = pd.Series(prices, dtype=float, index=self._horizon)
        # create an internal supplier node and add it to the internal data structure and the graph
        supplier = InternalSupplier(name=name, commodity=commodity, prices=prices, variance_fn=variance)
        self._check_and_update(node=supplier, parents=None, min_flow=None, max_flow=None, integer=None)
        return supplier.exposed

    def add_client(self,
                   name: str,
                   commodity: str,
                   parents: str | List[str],
                   demands: float | list | np.ndarray | pd.Series,
                   variance: Callable[[np.random.Generator, pd.Series], float] = lambda rng, series: 0.0) -> Client:
        """Adds a client node to the plant topology.

        :param name:
            The name of the client node.

        :param commodity:
            The identifier of the commodity that it requests, which must have been registered before.

        :param parents:
            The identifier of the parent nodes that are connected with the input of this client node.

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

        :return:
            The added client node.
        """
        # convert demands to a standard format (pd.Series)
        demands = pd.Series(demands, dtype=float, index=self._horizon)
        # create an internal client node and add it to the internal data structure and the graph
        client = InternalClient(name=name, commodity=commodity, demands=demands, variance_fn=variance)
        self._check_and_update(node=client, parents=parents, min_flow=0.0, max_flow=float('inf'), integer=False)
        return client.exposed

    def add_machine(self,
                    name: str,
                    commodity: str,
                    parents: str | List[str],
                    setpoint: Dict[str, Iterable[float]] | pd.DataFrame,
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
        machine = InternalMachine(
            name=name,
            setpoint=setpoint,
            commodity=commodity,
            discrete_setpoint=discrete_setpoint,
            max_starting=max_starting,
            cost=cost
        )
        self._check_and_update(node=machine, parents=parents, min_flow=min_flow, max_flow=max_flow, integer=integer)
        return machine.exposed

    def add_storage(self,
                    name: str,
                    commodity: str,
                    parents: str | List[str],
                    capacity: float = float('inf'),
                    dissipation: float = 0.0,
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

        :param min_flow:
            The minimal flow of commodity that can pass in the edge.

        :param max_flow:
            The maximal flow of commodity that can pass in the edge.

        :param integer:
            Whether the flow must be integer or not.

        :return:
            The added storage node.
        """
        # create an internal machine node and add it to the internal data structure and the graph
        storage = InternalStorage(name=name, commodity=commodity, capacity=capacity, dissipation=dissipation)
        self._check_and_update(node=storage, parents=parents, min_flow=min_flow, max_flow=max_flow, integer=integer)
        return storage.exposed

    def draw(self,
             figsize: Tuple[int, int] = (16, 9),
             node_colors: None | str | Dict[str, str] = None,
             node_markers: None | str | Dict[str, str] = None,
             edge_colors: str | Dict[str, str] = 'black',
             edge_shapes: str | Dict[str, str] = 'solid',
             node_size: float = 30,
             edge_width: float = 2,
             legend: bool = True):
        """Draws the plant topology.

        :param figsize:
            The matplotlib figsize parameter.

        :param node_colors:
            Either a string representing the color of the nodes, or a dictionary {kind: color} which associates a color
            to each node kind ('supplier', 'client', 'machine').

        :param node_markers:
            Either a string representing the shape of the nodes, or a dictionary {kind: shape} which associates a shape
            to each node kind ('supplier', 'client', 'machine', 'storage').

        :param edge_colors:
            Either a string representing the color of the edges, or a dictionary {commodity: color} which associates a
            color to each commodity that flows in an edge.

        :param edge_shapes:
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
        pos = drawing.get_node_positions(graph=graph, sources=nodes['supplier'].keys())
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
        styles = drawing.get_edge_style(colors=edge_colors, shapes=edge_shapes, commodities=list(self._commodities))
        for commodity, edge_list in self.edges().groupby('commodity'):
            commodity = str(commodity)
            drawing.draw_edges(
                graph=graph,
                pos=pos,
                edges=edge_list[['source', 'destination']].values,
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
