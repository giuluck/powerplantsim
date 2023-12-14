from typing import Set, Iterable, Any, Union

import networkx as nx
import numpy as np
import pandas as pd

from ppsim.datatypes import Edge, Node, Machine, Storage, Customer, \
    Purchaser, Supplier
from ppsim.utils import EdgeID
from ppsim.utils.typing import Plan, Flows


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


def process_plan(plan: Union[Plan, pd.DataFrame], edges: Set[EdgeID], horizon: pd.Index) -> pd.DataFrame:
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


# noinspection PyUnusedLocal
def default_action(index: Any, graph: nx.DiGraph) -> Flows:
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


def run_update(nodes: Iterable[Node], edges: Iterable[Edge], flows: Flows, rng: np.random.Generator):
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
        edge.update(flow=flows[edge.key])
    for node in nodes:
        node.update(rng=rng)


def build_output(nodes: Iterable[Node], edges: Iterable[Edge], horizon: pd.Index) -> SimulationOutput:
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
        output.flows[edge.key] = edge.flows
    for node in nodes:
        if isinstance(node, Machine):
            output.states[node.name] = node.states
        elif isinstance(node, Storage):
            output.storage[node.name] = node.storage
        elif isinstance(node, Customer):
            output.demands[node.name] = node.values
        elif isinstance(node, Purchaser):
            output.buying_prices[node.name] = node.values
        elif isinstance(node, Supplier):
            output.sell_prices[node.name] = node.values
        else:
            raise AssertionError(f"Unknown node type {type(node)}")
    return output
