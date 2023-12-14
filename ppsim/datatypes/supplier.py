from dataclasses import dataclass
from typing import Set, Optional, List

import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import VarianceNode


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class Supplier(VarianceNode):
    """A node in the plant that can supply a unique commodity."""

    @classproperty
    def kind(self) -> str:
        return 'supplier'

    @classproperty
    def _properties(self) -> List[str]:
        properties = super(VarianceNode, self)._properties
        return properties + ['predicted_prices', 'prices']

    @property
    def predicted_prices(self) -> pd.Series:
        """The series of predicted selling prices."""
        return self.predictions

    @property
    def prices(self) -> pd.Series:
        """The series of actual selling prices, which is filled during the simulation."""
        return self.values

    @property
    def commodity_in(self) -> Optional[str]:
        return None

    @property
    def commodities_out(self) -> Set[str]:
        return {self.commodity}
