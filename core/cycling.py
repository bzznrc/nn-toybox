"""Small helpers for rotating UI choices."""

from __future__ import annotations

from collections.abc import Sequence


def cycle_value(values: Sequence[str], current: object, delta: int = 1) -> str:
    if not values:
        raise ValueError("cycle_value needs at least one value")
    current_text = str(current)
    index = values.index(current_text) if current_text in values else 0
    return str(values[(index + int(delta)) % len(values)])
