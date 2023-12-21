from typing import Iterable, Union, Sized, Dict, Tuple

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
                 machines: Dict[NodeID, Machine],
                 edges: Dict[EdgeID, Edge],
                 horizon: pd.Index) -> pd.DataFrame:
    """Checks that the given plan is correctly specified and converts it to a standard format.

    :param plan:
        The energetic plan of the power plant defined as vectors of states/flows.

    :param machines:
        The dictionary of all the machine nodes in the plant for which a vector of states must be provided.

    :param edges:
        The dictionary of all the edges in the plant for which a vector of flows must be provided.

    :param horizon:
        The time horizon of the simulation.

    :return:
        The energetic plan in standard format.
    """
    # convert dictionary to dataframe if needed, and check consistency of flow vectors
    if isinstance(plan, dict):
        df = pd.DataFrame(index=horizon)
        for datatype, vector in plan.items():
            assert not isinstance(vector, Sized) or len(vector) == len(horizon), \
                f"Vector for key '{datatype}' has length {len(vector)}, expected {len(horizon)}"
            df[datatype] = pd.Series(vector, index=horizon)
        plan = df
    # distinguish between states and flows while checking that all the datatypes (columns) are in the given sets
    # and return a concatenated version of the states and flows dataframes with a higher level column index
    states, flows = check_plan(plan=plan, machines=machines, edges=edges)
    s, f = pd.DataFrame(states, index=horizon), pd.DataFrame(flows, index=horizon)
    # this is needed to avoid multi-column index
    f.columns = [k for k in flows.keys()]
    return pd.concat((s, f), keys=['states', 'flows'], axis=1)


def check_plan(plan: Union[Plan, pd.DataFrame], machines: Dict[NodeID, Machine], edges: Dict[EdgeID, Edge]) -> \
        Tuple[Dict[NodeID, Union[float, Iterable[float]]], Dict[Tuple[str, str, str], Union[float, Iterable[float]]]]:
    """Checks that the given plan is correctly specified and returns the dictionaries of states and flows.

    :param plan:
        The energetic plan of the power plant defined as vectors of states/flows.

    :param machines:
        The dictionary of all the machine nodes in the plant for which a vector of states must be provided.

    :param edges:
        The dictionary of all the edges in the plant for which a vector of flows must be provided.

    :return:
        A tuple (states, flows), where state is a dictionary <node: state/states> and flows is a dictionary
        <(source, destination, commodity): flow/flows>.
    """
    machines, edges = machines.copy(), edges.copy()
    states, flows = {}, {}
    for key, value in plan.items():
        if key in machines:
            machines.pop(key)
            states[key] = value
        elif key in edges:
            edge = edges.pop(key)
            flows[key[0], key[1], edge.commodity] = value
        else:
            raise AssertionError(f"Key {utils.stringify(key)} is not present in the plant")
    # check that the remaining sets are empty, i.e., there is no missing states/flows in the dataframe
    assert len(machines) == 0, f"No states vector has been passed for machines {list(machines.keys())}"
    assert len(edges) == 0, f"No flows vector has been passed for edges {list(edges.keys())}"
    return states, flows


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
        output.flows[edge.key[0], edge.key[1], edge.commodity] = edge.flows
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
