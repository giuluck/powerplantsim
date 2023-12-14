from dataclasses import dataclass, field
from typing import Set, Optional, Tuple, List

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import Node


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Machine(Node):
    """A node in the plant that converts certain commodities in other."""

    commodity: str = field(kw_only=True)
    """The input commodity of the machine."""

    _setpoint: pd.DataFrame = field(kw_only=True)
    """A pandas dataframe where the index is a series of floating point values that indicate the input commodity flow,
    while the columns should be named after the output commodity and contain the respective output flows."""

    discrete_setpoint: bool = field(kw_only=True)
    """Whether the setpoint is discrete or continuous."""

    max_starting: Optional[Tuple[int, int]] = field(kw_only=True)
    """A tuple <N, T> where N is the maximal number of times that the machine can be switched on in T units."""

    cost: float = field(kw_only=True)
    """The cost for operating the machine (cost is discarded when the machine is off)."""

    _states: List[float] = field(init=False, default_factory=list)
    """The series of actual input setpoints (None for machine off), which is filled during the simulation."""

    def __post_init__(self):
        # sort setpoint and rename index
        self._setpoint.sort_index(inplace=True)
        self._setpoint.index.rename(name='setpoint', inplace=True)
        # check that set points are strictly positive (the first one is enough since it is sorted)
        assert self._setpoint.index[0] >= 0.0, f"Setpoints should be non-negative, got {self._setpoint.index[0]}"
        # check that the corresponding output flows are non-negative (take the minimum value for each column)
        for c in self._setpoint.columns:
            lb = self._setpoint[c].min()
            assert lb >= 0.0, f"Output flows should be non-negative, got {c}: {lb}"
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
        return properties + ['setpoint', 'discrete_setpoint', 'max_starting', 'cost', 'states']

    @property
    def states(self) -> pd.Series:
        """The series of actual input setpoints (NaN for machine off), which is filled during the simulation."""
        return pd.Series(self._states, dtype=float, index=self._horizon[:self._info.step + 1])

    @property
    def setpoint(self) -> pd.DataFrame:
        """A pandas dataframe where the index is a series of floating point values that indicate the input commodity
        flow, while the columns should be named after the output commodity and contain the respective output flows."""
        return self._setpoint.copy()

    @property
    def commodity_in(self) -> Optional[str]:
        return self.commodity

    @property
    def commodities_out(self) -> Set[str]:
        return set(self._setpoint.columns)

    def update(self, rng: np.random.Generator):
        step = self._step()
        setpoint = self._setpoint.index
        # compute total input and output flows from respective edges
        in_flow = np.sum([e.flow_at(step=step) for e in self._in_edges])
        if in_flow == 0.0:
            # zero outputs for machine off and input flow becomes None as it will be stored in the series of setpoints
            in_flow = None
            out_flows = {col: 0.0 for col in self._setpoint.columns}
        elif self.discrete_setpoint:
            # check correctness of discrete setpoint and compute output
            assert in_flow in setpoint, \
                f"Unsupported input flow {in_flow} for discrete setpoint {list(setpoint)} in machine '{self.name}'"
            out_flows = self._setpoint.loc[in_flow].to_dict()
        else:
            # check correctness of continuous setpoint and compute output by interpolating the flow over the commodities
            lb, ub = setpoint[[0, -1]]
            assert lb <= in_flow <= ub, \
                f"Unsupported flow {in_flow} for continuous setpoint {lb} <= flow <= {ub} in machine '{self.name}'"
            out_flows = {col: np.interp(in_flow, xp=setpoint, fp=self._setpoint[col]) for col in self._setpoint.columns}
        # check that the respective output flows are consistent
        out_true = {col: 0.0 for col in self._setpoint.columns}
        for e in self._out_edges:
            out_true[e.commodity] += e.flow_at(step=step)
        for commodity, exp_flow in out_flows.items():
            true_flow = out_true[commodity]
            assert np.isclose(true_flow, exp_flow), \
                f"Expected output flow {exp_flow} for commodity '{commodity}' in machine '{self.name}', got {true_flow}"
        # check maximal number of starting:
        #  - create a list of the last T states and append the last one
        #  - check consecutive pairs and increase the counter if we pass from a NaN to a real number
        #  - if the counter gets greater than N, raise an error
        if self.max_starting is not None:
            n, t = self.max_starting
            t = min(t, len(self._states))
            count = 0
            states = [*self._states.values[-t:], in_flow]
            for s1, s2 in zip(states[-t - 1:-1], states[-t:]):
                if np.isnan(s1) and not np.isnan(s2):
                    count += 1
                    assert count <= n, \
                        f"Machine '{self.name}' cannot be started for more than {n} times in {t} time steps"
        self._states[step] = in_flow
