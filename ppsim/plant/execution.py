from dataclasses import field, dataclass
from typing import Union, Sized, Dict, List, Tuple, Iterable

import pandas as pd

from ppsim import utils
from ppsim.datatypes import Machine, Storage, Customer, Purchaser, Supplier, Edge, Node
from ppsim.utils.typing import NodeID, Plan, StepPlan, State, Flow, EdgeID, NamedTuple

States = Dict[Machine, Union[State, pd.Series]]
"""Datatype for states."""

Flows = Dict[Edge, Union[Flow, pd.Series]]
"""Datatype for flows."""


@dataclass(frozen=True, unsafe_hash=True, slots=True)
class StepPlanInfo(NamedTuple):
    """Datatype for the definition of the plan for a single step."""

    states: Dict[Machine, State] = field(default_factory=dict, init=False)
    """The states for the given step."""

    flows: Dict[Edge, Flow] = field(default_factory=dict, init=False)
    """The flows for the given step."""


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


def check_plant(plant):
    """Checks that the plant is valid, i.e., that each commodity flows correctly and has a start and end point.
    If the plant is not valid, raises an exception.

    :param plant:
        The plant to check.
    """
    for name, node in plant.nodes().items():
        for commodity in node.commodities_in:
            edges = plant.edges(destinations=name, commodities=commodity)
            assert len(edges) > 0, f"Input commodity {commodity} has no valid ingoing edge in node {name}"
        for commodity in node.commodities_out:
            edges = plant.edges(sources=name, commodities=commodity)
            assert len(edges) > 0, f"Output commodity {commodity} has no valid outgoing edge in node {name}"


# noinspection PyTypeChecker
def process_plan(plan: Union[Plan, pd.DataFrame],
                 machines: Dict[NodeID, Machine],
                 edges: Dict[EdgeID, Edge],
                 horizon: pd.Index) -> Tuple[List[StepPlanInfo], Dict[NodeID, pd.Series], Dict[EdgeID, pd.Series]]:
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
        A tuple where the first element is the energetic plan as a list of step plans, while the remaining two elements
        are dictionaries of states/flows indexed by nodes/edges.
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
    output = [StepPlanInfo() for _ in range(len(plan))]
    for machine, values in states.items():
        for i, value in enumerate(values):
            output[i].states[machine] = value
    for edge, values in flows.items():
        for i, value in enumerate(values):
            output[i].flows[edge] = value
    return output, states, flows


def check_plan(plan: Union[StepPlan, pd.Series, Plan, pd.DataFrame],
               machines: Dict[NodeID, Machine],
               edges: Dict[EdgeID, Edge]) -> Tuple[States, Flows]:
    """Checks that the given plan is correctly specified and returns the dictionaries of states and flows.

    :param plan:
        The energetic plan of the power plant defined as vectors of states/flows.

    :param machines:
        The dictionary of all the machine nodes in the plant for which a vector of states must be provided.

    :param edges:
        The dictionary of all the edges in the plant for which a vector of flows must be provided.

    :return:
        A tuple (states, flows) containing the dictionaries of states/flows indexed by machine/edge.
    """
    machines, edges = machines.copy(), edges.copy()
    states, flows = {}, {}
    for key, value in plan.items():
        if key in machines:
            mac = machines.pop(key)
            states[mac] = value
        elif key in edges:
            edge = edges.pop(key)
            flows[edge] = value
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
