from dataclasses import dataclass
from typing import Set, Optional

import pandas as pd
from descriptors import classproperty

from ppsim.datatypes.node import InternalVarianceNode, VarianceNode


@dataclass(repr=False, eq=False, slots=True)
class Supplier(VarianceNode):
    """A node in the plant that can supply a unique commodity and can be exposed to the user."""

    @property
    def prices(self) -> pd.Series:
        """The series of predicted selling prices."""
        return self.predictions


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, kw_only=True, slots=True)
class InternalSupplier(InternalVarianceNode):
    """A node in the plant that can supply a unique commodity and is not exposed to the user."""

    @classproperty
    def kind(self) -> str:
        return 'supplier'

    @property
    def commodity_in(self) -> Optional[str]:
        return None

    @property
    def commodities_out(self) -> Set[str]:
        return {self.commodity}

    @property
    def exposed(self) -> Supplier:
        return Supplier(
            name=self.name,
            commodity_in=self.commodity_in,
            commodities_out=self.commodities_out,
            predictions=self.predictions.copy(deep=True)
        )
