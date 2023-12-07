from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Tuple

import numpy as np
import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import Node


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False)
class Machine(Node):
    """A node in the plant that converts certain commodities in other."""

    commodity: str = field(kw_only=True)
    """The input commodity of the machine."""

    setpoint: pd.DataFrame = field(kw_only=True)
    """A pandas dataframe where the index is a series of floating point values that indicate the input commodity flow,
    while the columns should be named after the output commodity and contain the respective output flows."""

    discrete_setpoint: bool = field(kw_only=True)
    """Whether the setpoint is discrete or continuous."""

    max_starting: Optional[Tuple[int, int]] = field(kw_only=True)
    """A tuple <N, T> where N is the maximal number of times that the machine can be switched on in T units."""

    cost: float = field(kw_only=True)
    """The cost for operating the machine (cost is discarded when the machine is off)."""

    def __post_init__(self):
        # sort setpoint
        self.setpoint.sort_index(inplace=True)
        # check that set points are non-negative (the first one is enough since it is sorted)
        assert self.setpoint.index[0] > 0.0, f"Setpoints should be strictly positive, got {self.setpoint.index[0]}"
        # check non-negative cost
        assert self.cost >= 0.0, f"The operating cost of the machine must be non-negative, got {self.cost}"

    @classproperty
    def kind(self) -> str:
        return 'machine'

    @classproperty
    def commodity_in(self) -> bool:
        return True

    @classproperty
    def commodity_out(self) -> bool:
        return True

    @property
    def commodities_in(self) -> Set[str]:
        return {self.commodity}

    @property
    def commodities_out(self) -> Set[str]:
        return set(self.setpoint.columns)

    def operate(self, flow: float) -> Optional[Dict[str, float]]:
        """Converts a dictionary of input commodities into a dictionary of output commodities.

        :param flow:
            The flow of input commodities.

        :return:
            A dictionary the specifies the quantity of output commodities, indexed by name.
            If the input flow is null (0.0), the we consider that the machine is off, hence None is returned.
        """
        # return None for machine off
        if flow == 0.0:
            return None
        index = self.setpoint.index
        # check correctness of discrete setpoint and return output
        if self.discrete_setpoint:
            assert flow in index, f"Unsupported flow {flow} for discrete setpoint {list(index)}"
            return self.setpoint.loc[flow].to_dict()
        # check correctness of continuous setpoint and return output by interpolating the flow over the output column
        minimum, maximum = index[[0, -1]]
        assert minimum <= flow <= maximum, f"Unsupported flow {flow} for discrete setpoint {minimum} < flow < {maximum}"
        return {col: np.interp(flow, xp=index, fp=self.setpoint[col]) for col in self.setpoint.columns}
