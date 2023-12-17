from dataclasses import dataclass, field
from typing import Set, Optional, Tuple, List

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim import utils
from ppsim.datatypes.node import Node
from ppsim.utils.typing import State, Flows, States


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Machine(Node):
    """A node in the plant that converts certain commodities in other."""

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
        return properties + ['setpoint', 'discrete_setpoint', 'max_starting', 'cost', 'states', 'current_state']

    @property
    def states(self) -> pd.Series:
        """The series of actual input setpoints (NaN for machine off), which is filled during the simulation."""
        return pd.Series(self._states, dtype=float, index=self._horizon[:len(self._states)])

    @property
    def current_state(self) -> Optional[State]:
        """The current state of the machine for this time step as provided by the user."""
        return self._info['current_state']

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

    def update(self, rng: np.random.Generator, flows: Flows, states: States):
        self._info['current_state'] = states[self.key]

    def step(self, flows: Flows, states: States):
        state = states[self.key]
        # compute total input and output flows from respective edges
        machine_flows = {('input', commodity): 0.0 for commodity in self.commodities_in}
        machine_flows.update({('output', commodity): 0.0 for commodity in self.commodities_out})
        for (source, destination, commodity), flow in flows.items():
            if source == self.name:
                machine_flows[('output', commodity)] += flow
            if destination == self.name:
                machine_flows[('input', commodity)] += flow
        # if the flow is 0.0 or nan, return nan state (off), check that the input/output flows are null
        if state is None or np.isnan(state):
            for (key, commodity), flow in machine_flows.items():
                assert np.isclose(flow, 0.0), \
                    f"Got non-zero {key} flow for '{commodity}' despite null setpoint for machine '{self.name}'"
            self._states.append(state)
            return
        # if discrete setpoint
        #  - check that the given state is valid
        #  - check that the flows match the given state
        if self.discrete_setpoint:
            assert state in self._setpoint.index, f"Unsupported state {state} for machine '{self.name}'"
            for (key, commodity), flow in machine_flows.items():
                expected = self._setpoint[(key, commodity)].loc[state]
                assert np.isclose(expected, flow), \
                    f"Flow {expected} expected for machine '{self.name}' with state {state}, got {flow}"
        # if continuous setpoint:
        #  - check that the given state is within the expected bounds
        else:
            lb, ub = self._setpoint.index[[0, -1]]
            assert lb <= state <= ub, f"Unsupported state {state} for machine '{self.name}'"
            for (key, commodity), flow in machine_flows.items():
                expected = np.interp(state, xp=self._setpoint.index, fp=self._setpoint[(key, commodity)])
                assert np.isclose(expected, flow), \
                    f"Expected flow {expected} for {key} commodity '{commodity}' in machine '{self.name}', got {flow}"
        # check maximal number of starting:
        #  - create a list of the last T states, prepend nan (machine starts off) and append the last one
        #  - check consecutive pairs and increase the counter if we pass from a NaN to a real number
        #  - if the counter gets greater than N, raise an error
        if self.max_starting is not None:
            n, t = self.max_starting
            t = min(t, len(self._states) + 1)
            count = 0
            states = [np.nan, *self._states[-t:], state]
            for s1, s2 in zip(states[-t - 1:-1], states[-t:]):
                if (s1 is None or np.isnan(s1)) and not (s2 is None or np.isnan(s2)):
                    count += 1
                    assert count <= n, f"Machine '{self.name}' cannot be started for more than {n} times in {t} steps"
        self._states.append(state)
        self._info['current_state'] = None
