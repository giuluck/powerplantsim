from dataclasses import dataclass
from typing import Set, List, Optional

import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import VarianceNode
from ppsim.utils.typing import Flows, States


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Supplier(VarianceNode):
    """A node in the plant that can supply a unique commodity."""

    @classproperty
    def kind(self) -> str:
        return 'supplier'

    @classproperty
    def _properties(self) -> List[str]:
        properties = super(Supplier, self)._properties
        return properties + ['prices', 'current_price']

    @property
    def prices(self) -> pd.Series:
        """The series of actual selling prices, which is filled during the simulation."""
        return self.values

    @property
    def current_price(self) -> Optional[float]:
        """The current selling price of the node for this time step as computed using the variance model."""
        return self.current_value

    @property
    def commodities_in(self) -> Set[str]:
        return set()

    @property
    def commodities_out(self) -> Set[str]:
        return {self.commodity}

    def step(self, flows: Flows, states: States):
        self._values.append(self._info['current_value'])
        self._info['current_value'] = None
