"""Named registries: the extension points a branch uses to add a loss, a head, a data transform, an encoding, a
backbone or an attention function without touching the training loop.

    from zlab.losses import LOSSES

    @LOSSES.register("kl_anchor")
    class KLAnchor(LossTerm): ...

and then name it in `bench/variant.json`. Registration happens at import time, so a branch puts its code into an
existing module (or a new module listed in `variant.json` under "plugins", imported before the model is built).
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items and self._items[name] is not obj:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} '{name}'; registered: {sorted(self._items)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)
