from typing import Dict, Tuple, Set, Iterable, Any

import networkx as nx
import numpy as np
import pandas as pd

from ppsim.datatypes import InternalEdge, InternalNode, InternalMachine, InternalStorage, InternalCustomer, \
    InternalPurchaser, InternalSupplier


class SimulationOutput:
    """Dataclass representing the output of a power plant simulation."""

    def __init__(self, horizon: pd.Index):
        """
        :param horizon:
            The time horizon of the simulation.
        """
        self.flows: pd.DataFrame = pd.DataFrame(index=horizon, dtype=float)
        """The commodities flows indexed by edge (source, destination)."""

        self.states: pd.DataFrame = pd.DataFrame(index=horizon, dtype=float)
        """The machine setpoints indexed by machine name."""

        self.storage: pd.DataFrame = pd.DataFrame(index=horizon, dtype=float)
        """The commodities that were stored in the plant indexed by storage name."""

        self.demands: pd.DataFrame = pd.DataFrame(index=horizon, dtype=float)
        """The true demands indexed by customer name."""

        self.buying_prices: pd.DataFrame = pd.DataFrame(index=horizon, dtype=float)
        """The true buying prices indexed by purchaser name."""

        self.sell_prices: pd.DataFrame = pd.DataFrame(index=horizon, dtype=float)
        """The true selling prices indexed by supplier name."""


def process_plan(plan: Dict[Tuple[str, str], float | Iterable[float]] | pd.DataFrame,
                 edges: Set[Tuple[str, str]],
                 horizon: pd.Index) -> pd.DataFrame:
    """Checks that the given plan is correctly specified and converts it to a standard format.

    :param plan:
        The energetic plan of the power plant defined as vectors of flows within the edges of the power plant.

    :param edges:
        The set of all the edges in the plant for which a vector of flows must be provided.

    :param horizon:
        The time horizon of the simulation.

    :return:
        The energetic plan in standard format.
    """
    # convert dictionary to dataframe if needed, and check consistency of flow vectors
    if isinstance(plan, dict):
        df = pd.DataFrame(index=horizon)
        for edge, flows in plan.items():
            assert len(flows) == len(horizon), f"Flows for edge {edge} has length {len(flows)}, expected {len(horizon)}"
            df[edge] = pd.Series(flows, index=horizon)
        plan = df
    # check that all the edges (columns) are in the edge set and pop them
    for edge in plan.columns:
        assert edge in edges, f"Edge {edge} is not present in the plant"
        edges.remove(edge)
    # check that the remaining edge set is empty, i.e., there is no missing edge in the dataframe
    assert len(edges) == 0, f"No flows vector has been passed for edges {edges}"
    return plan


def action_graph(plant, flows: pd.Series) -> nx.DiGraph:
    """Builds a graph representing the power plant, with flows included as attributes on edges.

    :param plant:
        The power plant object.

    :param flows:
        The series of edge flows indexed by edge.

    :return:
        The graph instance.
    """
    graph = plant.graph(attributes=True)
    for (source, destination), flow in flows.items():
        graph[source][destination]['flow'] = flow
    return graph


# noinspection PyUnusedLocal
def default_action(index: Any, graph: nx.DiGraph) -> Dict[Tuple[str, str], float]:
    """Implements the default recourse action.

    :param index:
        The time index of the simulation.

    :param graph:
        A graph representing the topology of the power plant, with flows included as attributes on edges.

    :return:
        A dictionary {<edge>: <updated_flow>} where an <edge> is identified by the tuple of the names of the node that
        is connecting, and <updated_flow> is a floating point value of the actual flow.
    """
    # TODO: implement default action
    return {(source, destination): attributes['flow'] for source, destination, attributes in graph.edges(data=True)}


def run_update(nodes: Iterable[InternalNode],
               edges: Iterable[InternalEdge],
               flows: Dict[Tuple[str, str], float],
               rng: np.random.Generator):
    """Performs a step update in the simulation.

    :param nodes:
        The simulation nodes.

    :param edges:
        The simulation edges.

    :param flows:
        The dictionary of actual flows obtained from the recourse action.

    :param rng:
        The random number generator to be used for reproducible results.
    """
    for edge in edges:
        edge.update(flow=flows[(edge.source.name, edge.destination.name)])
    for node in nodes:
        node.update(rng=rng)


def build_output(nodes: Iterable[InternalNode], edges: Iterable[InternalEdge], horizon: pd.Index) -> SimulationOutput:
    """Builds the output of the simulation.

    :param nodes:
        The simulation nodes.

    :param edges:
        The simulation edges.

    :param horizon:
        The time horizon of the simulation.

    :return:
        A SimulationOutput object containing all the information about true prices, demands, setpoints, and storage.
    """
    output = SimulationOutput(horizon=horizon)
    for edge in edges:
        output.flows[(edge.source.name, edge.destination.name)] = edge.flows
    for node in nodes:
        if isinstance(node, InternalMachine):
            output.states[node.name] = node.states
        elif isinstance(node, InternalStorage):
            output.storage[node.name] = node.storage
        elif isinstance(node, InternalCustomer):
            output.demands[node.name] = node.values
        elif isinstance(node, InternalPurchaser):
            output.buying_prices[node.name] = node.values
        elif isinstance(node, InternalSupplier):
            output.sell_prices[node.name] = node.values
        else:
            raise AssertionError(f"Unknown node type {type(node)}")
    return output
