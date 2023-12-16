from typing import Iterable, Union, Sized

import networkx as nx
import pandas as pd

from ppsim import utils
from ppsim.datatypes import Edge, Node, Machine, Storage, Customer, Purchaser, Supplier
from ppsim.utils.typing import EdgeID, NodeID, Plan


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


def process_plan(plan: Union[Plan, pd.DataFrame],
                 machines: Iterable[NodeID],
                 edges: Iterable[EdgeID],
                 horizon: pd.Index) -> pd.DataFrame:
    """Checks that the given plan is correctly specified and converts it to a standard format.

    :param plan:
        The energetic plan of the power plant defined as vectors of states/flows.

    :param machines:
        The set of all the machine nodes in the plant for which a vector of states must be provided.

    :param edges:
        The set of all the edges in the plant for which a vector of flows must be provided.

    :param horizon:
        The time horizon of the simulation.

    :return:
        The energetic plan in standard format.
    """
    machines, edges = set(machines), set(edges)
    # convert dictionary to dataframe if needed, and check consistency of flow vectors
    if isinstance(plan, dict):
        df = pd.DataFrame(index=horizon)
        for datatype, vector in plan.items():
            assert not isinstance(vector, Sized) or len(vector) == len(horizon), \
                f"Vector for object '{datatype}' has length {len(vector)}, expected {len(horizon)}"
            df[datatype] = pd.Series(vector, index=horizon)
        plan = df
    # distinguish between states and flows while checking that all the datatypes (columns) are in the given sets
    states, flows = [], []
    for key in plan.columns:
        if key in machines:
            states.append(key)
            machines.remove(key)
        elif key in edges:
            flows.append(key)
            edges.remove(key)
        else:
            raise AssertionError(f"Key {utils.stringify(key)} is not present in the plant")
    # check that the remaining sets are empty, i.e., there is no missing states/flows in the dataframe
    assert len(machines) == 0, f"No states vector has been passed for machines {machines}"
    assert len(edges) == 0, f"No flows vector has been passed for edges {edges}"
    # return a concatenated version of the states and flows dataframes with a higher level column index
    return pd.concat((plan[states], plan[flows]), keys=['states', 'flows'], axis=1)


# noinspection PyUnusedLocal
def default_action(step: int, graph: nx.DiGraph) -> Plan:
    """Implements the default recourse action.

    :param step:
        The time step of the simulation.

    :param graph:
        A graph representing the topology of the power plant, with flows included as attributes on edges.

    :return:
        A dictionary {machine | edge: updated_state | updated_flow} where a machine is identified by its name and an
        edge is identified by the tuple of the names of the node that is connecting, while updated_state and
        updated_flow is the value of the actual state/flow.
    """
    # TODO: implement default action
    output = {}
    for name, attributes in graph.nodes(data=True):
        if attributes['kind'] == 'machine':
            output[name] = attributes['current_state']
    for source, destination, attributes in graph.edges(data=True):
        output[(source, destination)] = attributes['current_flow']
    return output


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
