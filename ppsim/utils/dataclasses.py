from dataclasses import dataclass
from typing import Any, Dict, Tuple, List


@dataclass(frozen=True, unsafe_hash=True, slots=True)
class NamedTuple:
    def __getitem__(self, item: Any) -> Any:
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
