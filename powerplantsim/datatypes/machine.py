from dataclasses import dataclass, field
from typing import Set, Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
# noinspection PyPackageRequirements
from descriptors import classproperty
from pyomo.core import Piecewise

from powerplantsim import utils
from powerplantsim.datatypes.node import Node
from powerplantsim.utils.typing import State, Flow


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True)
class Machine(Node):
    """A node in the plant that converts certain commodities in others."""

    _setpoint: pd.DataFrame = field(kw_only=True)
    """A pandas dataframe where the index is a series of floating point values that indicate the input commodity flow,
    while the columns should be named after the output commodity and contain the respective output flows."""

    discrete_setpoint: bool = field(kw_only=True)
    """Whether the setpoint is discrete or continuous."""

    max_starting: Optional[Tuple[int, int]] = field(kw_only=True)
    """A tuple <N, T> where N is the maximal number of times that the machine can be switched on in T units."""

    cost: float = field(kw_only=True)
    """The cost for operating the machine (cost is discarded when the machine is off)."""

    _states: List[State] = field(init=False, default_factory=list)
    """The series of actual input setpoints (None for machine off), which is filled during the simulation."""

    def __post_init__(self):
        self._info['current_state'] = None
        # sort setpoint and rename index
        self._setpoint.sort_index(inplace=True)
        self._setpoint.index.rename(name='setpoint', inplace=True)
        # check that set points are strictly positive (the first one is enough since it is sorted)
        assert self._setpoint.index[0] >= 0.0, f"Setpoints should be non-negative, got {self._setpoint.index[0]}"
        for c in self._setpoint.columns:
            # check that the column is structured as a tuple ('input'|'output', commodity)
            assert isinstance(c, tuple) and len(c) == 2 and c[0] in ['input', 'output'], \
                f"Setpoint columns should be a tuple ('input'|'output', commodity), got {utils.stringify(c)}"
            # check that the corresponding output flows are non-negative (take the minimum value for each column)
            lb = self._setpoint[c].min()
            assert lb >= 0.0, f"Setpoint flows should be non-negative, got {c}: {lb}"
        # check max starting
        if self.max_starting is not None:
            n, t = self.max_starting
            assert n > 0, f"The number of starting must be a strictly positive integer, got {n}"
            assert n < t, f"The number of starting must be strictly less than the number of time steps, got {n} >= {t}"
        # check non-negative cost
        assert self.cost >= 0.0, f"The operating cost of the machine must be non-negative, got {self.cost}"

    @classproperty
    def kind(self) -> str:
        return 'machine'

    @classproperty
    def _properties(self) -> List[str]:
        properties = super(Machine, self)._properties
        return properties + [
            'commodities_in',
            'commodities_out',
            'setpoint',
            'discrete_setpoint',
            'max_starting',
            'cost',
            'states',
            'current_state'
        ]

    def starts(self, t: int) -> int:
        """Computes the number of times the machine has been started in the past <t> steps.

        :param t:
            The number of past steps to consider.

        :return:
            The number of machine starts.
        """
        if t <= 0:
            return 0
        count = 0
        # create a list of the last T states
        t = min(t, len(self._states))
        # prepend nan (machine starts off) and append the last one
        states = [np.nan, *self._states[-t:]]
        # check consecutive pairs and increase the counter if we pass from a NaN to a real number
        for s1, s2 in zip(states[:-1], states[1:]):
            if np.isnan(s1) and not np.isnan(s2):
                count += 1
        return count

    @property
    def states(self) -> pd.Series:
        """The series of actual input setpoints (NaN for machine off), which is filled during the simulation."""
        return pd.Series(self._states, dtype=float, index=self._horizon[:len(self._states)])

    @property
    def current_state(self) -> Optional[State]:
        """The current state of the machine for this time step as provided by the user."""
        return self._info['current_state']

    @property
    def previous_state(self) -> State:
        """The state of the machine in the previous time step (or np.nan if this is the first time step)."""
        return np.nan if len(self._states) == 0 else self._states[-1]

    @property
    def setpoint(self) -> pd.DataFrame:
        """A pandas dataframe where the index is a series of floating point values that indicate the input commodity
        flow, while the columns should be named after the output commodity and contain the respective output flows."""
        return self._setpoint.copy()

    @property
    def commodities_in(self) -> Set[str]:
        return set(self._setpoint['input'].columns)

    @property
    def commodities_out(self) -> Set[str]:
        return set(self._setpoint['output'].columns)

    # noinspection PyTypeChecker
    def to_pyomo(self, mutable: bool = False) -> pyo.Block:
        # start from the default node block and store aliases for setpoint lower and upper bounds
        node = super(Machine, self).to_pyomo(mutable=mutable)
        lb, ub = self._setpoint.index[[0, -1]]
        # add a parameter representing the current state (and initialize it if needed)
        current_state = self.current_state
        kwargs = dict(mutable=True) if mutable else dict(initialize=lb if np.isnan(current_state) else current_state)
        node.current_state = pyo.Param(domain=pyo.NonNegativeReals, **kwargs)
        # create a variable for the machine binary state (on/off) then, if the number of starts in the last <t - 1>
        # steps is already maximal, set the variable bounds as (0, 0) so that the machine will be forced to be off
        n, t = (1, 1) if self.max_starting is None else self.max_starting
        node.switch = pyo.Var(domain=pyo.Binary, bounds=(0, 1 if self.starts(t=t - 1) < n else 0), initialize=0)
        # compute the cost of operating the machine (in case it is on)
        node.cost = self.cost * node.switch
        # handle setpoint for discrete and continuous machines + trivial case with a unique setpoint
        kwargs = dict() if mutable else dict(initialize=lb if np.isnan(current_state) else current_state)
        if len(self._setpoint) <= 1:
            # the state is 0 if the machine is off, otherwise it is the unique state, same for the flows
            node.state = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, ub), **kwargs)
            node.state_cst = pyo.Constraint(rule=node.state == node.switch * self._setpoint.index[0])
            node.input_flows_cst = pyo.Constraint(
                node.in_flows.index_set(),
                rule=lambda _, com: node.in_flows[com] == node.switch * self._setpoint.input[com].iloc[0]
            )
            node.output_flows_cst = pyo.Constraint(
                node.out_flows.index_set(),
                rule=lambda _, com: node.out_flows[com] == node.switch * self._setpoint.output[com].iloc[0]
            )
        elif self.discrete_setpoint:
            # build a one-hot encoded selector that has a single entry if the machine is on (i.e., node.switch == 1)
            # or no entry if the machine is off (i.e., node.switch == 0)
            node.selector = pyo.Var(range(len(self._setpoint)), domain=pyo.Binary, initialize=0)
            node.selector_cst = pyo.Constraint(rule=sum(node.selector.values()) == node.switch)
            # build a variable for the actual setpoint so that it is equal to the value indexed by the selector
            #  - use a variable instead of a plain equation in order to access it via the ".value" property
            node.state = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, ub), **kwargs)
            node.state_cst = pyo.Constraint(rule=node.state == sum(node.selector * self._setpoint.index))
            # impose constraints on input/output flows so that they match the correct setpoint indexed by the selector
            node.input_flows_cst = pyo.Constraint(
                node.in_flows.index_set(),
                rule=lambda _, com: node.in_flows[com] == sum(node.selector * self._setpoint.input[com])
            )
            node.output_flows_cst = pyo.Constraint(
                node.out_flows.index_set(),
                rule=lambda _, com: node.out_flows[com] == sum(node.selector * self._setpoint.output[com])
            )
        else:
            # build a state variable that is bounded within the min and max setpoint
            breakpoints = self._setpoint.index.values
            node.state = pyo.Var(domain=pyo.NonNegativeReals, bounds=(lb, ub), **kwargs)
            # for each tuple of (input/output, commodity) flows:
            #  - build a flow variable which is bounded within the min and max flow
            #  - enforce a piecewise_linear constraint between the node state and the flow
            #  - constraint the respective flow "node.in_flows/out_flows[commodity]" to be node.switch * flow
            #    (i.e., if the machine is on then the final flow is equal to "flow", otherwise it is zero)
            for key, var in [('input', node.in_flows), ('output', node.out_flows)]:
                for commodity, values in self._setpoint[key].items():
                    # create the flow variable and add it to the node
                    v_min, v_max = values.min(), values.max()
                    flow = pyo.Var(domain=pyo.NonNegativeReals, bounds=(v_min, v_max))
                    node.add_component(f'{key}_{commodity}_flow', flow)
                    # create the piecewise linear constraint so that it:
                    #  - stores the output in the "flow" variable
                    #  - takes as input the "node.state" variable
                    #  - is defined via the setpoint index (breakpoints) and the respective flows (values)
                    #  - require tight equality bounds and is modelled using the SOS2 formulation
                    pwl_cst = Piecewise(
                        flow,
                        node.state,
                        pw_pts=list(breakpoints),
                        f_rule=list(values),
                        pw_constr_type='EQ',
                        pw_repn='SOS2'
                    )
                    node.add_component(f'{key}_{commodity}_pwl_cst', pwl_cst)
                    # constraint the final flow (var[commodity]) by linearizing the product node.switch * flow, i.e.:
                    #  - var[commodity] is upper-bounded by the actual value of the flow
                    #  - var[commodity] is greater than flow if node.switch = 1, otherwise the constraint is trivial
                    #  - var[commodity] is lower than zero if node.switch = 0, otherwise the constraint is trivial
                    flow_cst_ub = pyo.Constraint(rule=var[commodity] <= flow)
                    flow_cst_on = pyo.Constraint(rule=var[commodity] >= flow - (1 - node.switch) * v_max)
                    flow_cst_off = pyo.Constraint(rule=var[commodity] <= node.switch * v_max)
                    node.add_component(f'{key}_{commodity}_flow_cst_ub', flow_cst_ub)
                    node.add_component(f'{key}_{commodity}_flow_cst_on', flow_cst_on)
                    node.add_component(f'{key}_{commodity}_flow_cst_off', flow_cst_off)
        return node

    def update(self, rng: np.random.Generator, flows: Dict[Any, Flow], states: Dict[Any, State]):
        self._info['current_state'] = states[self]

    def step(self, flows: Dict[Any, Flow], states: Dict[Any, State]):
        state = states[self]
        # compute total input and output flows from respective edges
        machine_flows = {('input', commodity): 0.0 for commodity in self.commodities_in}
        machine_flows.update({('output', commodity): 0.0 for commodity in self.commodities_out})
        for edge, flow in flows.items():
            if edge.source == self.name:
                machine_flows[('output', edge.commodity)] += flow
            if edge.destination == self.name:
                machine_flows[('input', edge.commodity)] += flow
        # if the state is nan, check that the input/output flows are null
        if np.isnan(state):
            for (key, commodity), flow in machine_flows.items():
                assert np.isclose(flow, 0.0, atol=self.eps), \
                    f"Got non-zero {key} flow {flow} for '{commodity}' despite null setpoint for machine '{self.name}'"
            self._states.append(np.nan)
            return
        # if discrete setpoint
        #  - check that the given state is valid
        #  - check that the flows match the given state
        if self.discrete_setpoint:
            assert state in self._setpoint.index, f"Unsupported state {state} for machine '{self.name}'"
            for (key, commodity), flow in machine_flows.items():
                expected = self._setpoint[(key, commodity)].loc[state]
                assert np.isclose(expected, flow, rtol=self.eps), \
                    f"Flow {expected} expected for machine '{self.name}' with state {state}, got {flow}"
        # if continuous setpoint:
        #  - check that the given state is within the expected bounds
        else:
            lb, ub = self._setpoint.index[[0, -1]]
            assert lb - self.eps <= state <= ub + self.eps, f"Unsupported state {state} for machine '{self.name}'"
            state = np.clip(state, a_min=lb, a_max=ub)
            for (key, commodity), flow in machine_flows.items():
                expected = np.interp(state, xp=self._setpoint.index, fp=self._setpoint[(key, commodity)])
                assert np.isclose(expected, flow, rtol=self.eps), \
                    f"Expected flow {expected} for {key} commodity '{commodity}' in machine '{self.name}', got {flow}"
        # check maximal number of starting by checking that at least one of the following conditions is met:
        #  - the machine is off in this time step
        #  - the machine was on in the previous time step
        #  - the number of starts in the last <t - 1> steps is strictly lower than the maximal value required
        if self.max_starting is not None:
            n, t = self.max_starting
            assert np.isnan(state) or not np.isnan(self.previous_state) or self.starts(t=t - 1) < n, \
                f"Machine '{self.name}' cannot be started for more than {n} times in {t} steps"
        self._states.append(state)
        self._info['current_state'] = None
