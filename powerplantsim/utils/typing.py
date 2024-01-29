from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Iterable, Union

NodeID = str
"""Datatype for node identifier, i.e, its name."""

SingleEdgeID = Tuple[str, str]
"""Datatype for single-edge identifier, i.e., a tuple (source, destination)."""

MultiEdgeID = Tuple[str, str, str]
"""Datatype for multi-edge identifier, i.e., a tuple (source, destination, commodity)."""

EdgeID = SingleEdgeID
"""Alias for plant edges identifier, since plant do not support multi-graphs yet."""

State = float
"""Datatype for a single machine state specification (np.nan for machine off)."""

Flow = float
"""Datatype for a single edge flow specification."""

StepPlan = Dict[Union[NodeID, EdgeID], Union[State, Flow]]
"""Datatype for the specification of the plan of a single step, i.e., a dictionary <machine | edge: state | flow>."""

Plan = Dict[Union[NodeID, EdgeID], Union[State, Flow, Iterable[State], Iterable[Flow]]]
"""Datatype for specification of the total plan, i.e., a dictionary <machine | edge: state/states | flow/flows>."""


# noinspection PyUnresolvedReferences
@dataclass(frozen=True, unsafe_hash=True, slots=True)
class NamedTuple:
    """Template dataclass that models a named tuple."""

    def __getitem__(self, item: Any):
        if isinstance(item, int):
            item = self.__slots__[item]
        return getattr(self, item)

    def __len__(self):
        return len(self.__slots__)

    def __eq__(self, other: Any):
        # if another NamedTuple is passed, create
        if isinstance(other, NamedTuple):
            return self.dict == other.dict
        else:
            return self.tuple == other

    @property
    def dict(self) -> Dict[str, Any]:
        return {param: getattr(self, param) for param in self.__slots__}

    @property
    def list(self) -> List[Any]:
        return [getattr(self, param) for param in self.__slots__]

    @property
    def tuple(self) -> Tuple:
        return tuple(self.list)
