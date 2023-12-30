from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Union, Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from ppsim import utils
from ppsim.utils import NamedTuple
from ppsim.utils.typing import SimpleEdgeID


@dataclass(frozen=True, unsafe_hash=True, slots=True, kw_only=True)
class StyleInfo(NamedTuple):
    color: str = field(kw_only=True)
    shape: str = field(kw_only=True)


NODE_STYLES: Dict[str, StyleInfo] = {
    'supplier': StyleInfo(color='#FFEAB8', shape='>'),
    'customer': StyleInfo(color='#A6DEAE', shape='<'),
    'purchaser': StyleInfo(color='#FFEAB8', shape='<'),
    'machine': StyleInfo(color='#FF700A', shape='o'),
    'storage': StyleInfo(color='#ABA5E8', shape='s')
}
"""Default dictionary of nodes style information."""


def get_node_positions(graph: nx.DiGraph,
                       sources: Iterable[str],
                       node_pos: Union[str, List[Iterable[Optional[str]]], Dict[str, Any]]) -> dict:
    """Traverse the graph from the sources and label each node with a progressive number the position of the nodes is
    eventually obtained by layering them respectively to the computed number so that sources will be on the left and
    sinks on the right.

    :param graph:
        The networkx DiGraph instance.

    :param sources:
        The iterable of source nodes.

    :param node_pos:
        Information about the node position.

    :return:
        A dictionary of node positions.
    """
    # if the position info is a dictionary, it already contains the mapping <node: pos> so return the info itself
    if isinstance(node_pos, dict):
        return node_pos
    # if the position info is a list, create a new graph with nodes added sequentially and with the layers attribute
    if isinstance(node_pos, list):
        p = 0
        nodes = set(graph.nodes)
        edges = set(graph.edges)
        graph = nx.DiGraph()
        for layer, nodelist in enumerate(node_pos):
            for node in nodelist:
                if node is None:
                    p += 1
                    node = f'__placeholder-{p}__'
                else:
                    assert node in nodes, f"Node {node} is not present in the given plant"
                graph.add_node(node_for_adding=node, layer=layer)
        graph.add_edges_from(edges)
        pos = nx.multipartite_layout(graph, subset_key='layer')
        return {k: v for k, v in pos.items() if k in nodes}
    # if the position info is a string representing the layering strategy, add the layers to each node accordingly
    graph = graph.copy()
    if node_pos == 'sp':
        # use breadth first search for shortest paths
        for it, nodes in enumerate(nx.bfs_layers(graph, sources=sources)):
            for node in nodes:
                graph.nodes[node]['layer'] = it
    elif node_pos == 'lp':
        # use floyd warshall algorithm to search for longest paths
        #  - get the indices of the sources
        #  - get the negative shortest path matrix and select only the paths from the sources
        #  - get the negative minimum value for each node in the plant and negate it to get the layer
        nx.set_edge_attributes(graph, values=-1, name='weight')
        sources = set(sources)
        sources = [i for i, node in enumerate(graph.nodes) if node in sources]
        lp = -nx.floyd_warshall_numpy(graph)[sources].min(axis=0)
        for i, node in enumerate(graph.nodes):
            graph.nodes[node]['layer'] = lp[i]
    else:
        raise AssertionError(f"Unsupported node_pos: {node_pos}")
    return nx.multipartite_layout(graph, subset_key='layer')


def get_node_style(colors: Union[None, str, Dict[str, str]],
                   markers: Union[None, str, Dict[str, str]]) -> Dict[str, StyleInfo]:
    """Builds a dictionary of color and marker mappings indexed by node kind.
    In case either the colors or the shapes are not None, include the custom information in the dictionary.

    :param colors:
        Either a string representing the color of the nodes, or a dictionary {kind: color} which associates a color to
        each node kind ('supplier', 'client', 'machine').

    :param markers:
        Either a string representing the shape of the nodes, or a dictionary {kind: shape} which associates a shape to
        each node kind ('supplier', 'client', 'machine', 'storage').

    :return:
        The dictionary of style information.
    """
    styles = {}
    for kind, style in NODE_STYLES.items():
        color = utils.get_matching_object(matcher=colors, index=kind, default=style.color)
        marker = utils.get_matching_object(matcher=markers, index=kind, default=style.shape)
        styles[kind] = StyleInfo(color=color, shape=marker)
    return styles


def get_edge_style(colors: Union[None, str, Dict[str, str]],
                   shapes: Union[None, str, Dict[str, str]],
                   commodities: List[str]) -> Dict[str, StyleInfo]:
    """Build a dictionary of color and style mappings indexed by commodity.

    :param colors:
        Either a string representing the color of the commodity, or a dictionary {kind: color}.

    :param shapes:
        Either a string representing the shape of the commodity, or a dictionary {kind: shape}.

    :param commodities:
        The list of all the commodities in the plant.

    :return:
        The dictionary of style information.
    """
    styles = {}
    for com in commodities:
        color = utils.get_matching_object(matcher=colors, index=com, default='black')
        marker = utils.get_matching_object(matcher=shapes, index=com, default='solid')
        styles[com] = StyleInfo(color=color, shape=marker)
    return styles


def build_node_label(kind: str, style: StyleInfo) -> Line2D:
    """Create a label for the legend of nodes with the correct style, color, and text.

    :param kind:
        The type of nodes.

    :param style:
        The style information.

    :return:
        A Line2D object representing the label.
    """
    return Line2D(
        [],
        [],
        marker=style.shape,
        markerfacecolor=style.color,
        markeredgecolor='black',
        linestyle='None',
        markersize=25,
        label=kind.title()
    )


def build_edge_label(commodity: str, style: StyleInfo) -> Line2D:
    """Create a label for the legend of edges with the correct style, color, and text.

    :param commodity:
        The type of commodity flowing in the edge.

    :param style:
        The style information.

    :return:
        A Line2D object representing the label.
    """
    return Line2D(
        [],
        [],
        lw=2,
        color=style.color,
        linestyle=style.shape,
        label=commodity.title()
    )


def draw_nodes(graph: nx.DiGraph,
               pos: dict,
               nodes: Iterable[str],
               style: StyleInfo,
               size: float,
               width: float,
               ax: plt.Axes):
    """Draws a subset of nodes from the plant.

    :param graph:
        The networkx DiGraph instance.

    :param pos:
        The dictionary of node's positions.

    :param nodes:
        The subset of nodes to be drawn.

    :param style:
        The style information.

    :param size:
        The size of the nodes.

    :param width:
        The width of the node's borders.

    :param ax:
        The ax on which to plot.
    """
    nx.draw(
        graph,
        pos=pos,
        edgelist=[],
        nodelist=nodes,
        node_color=style.color,
        node_shape=style.shape,
        node_size=size * 100,
        linewidths=width,
        with_labels=True,
        edgecolors='k',
        arrows=True,
        ax=ax
    )


def draw_edges(graph: nx.DiGraph,
               pos: dict,
               edges: Iterable[SimpleEdgeID],
               style: StyleInfo,
               size: float,
               width: float,
               ax: plt.Axes):
    """Draws a subset of edges from the plant.

    :param graph:
        The networkx DiGraph instance.

    :param pos:
        The dictionary of node's positions.

    :param edges:
        The subset of edges to be drawn.

    :param style:
        The style information.

    :param size:
        The size of the nodes.

    :param width:
        The width of the edges.

    :param ax:
        The ax on which to plot.
    """
    nx.draw(
        graph,
        pos=pos,
        nodelist=[],
        edgelist=edges,
        edge_color=style.color,
        style=style.shape,
        node_size=size * 100,
        arrowsize=width * 10,
        width=width,
        arrows=True,
        ax=ax
    )
