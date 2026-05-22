"""Tiny hand-written semantic corpus."""

from __future__ import annotations

import itertools
import random


TINY_SEMANTICS = (
    ("cat", "has", "fur"),
    ("dog", "has", "fur"),
    ("bird", "has", "wings"),
    ("fish", "swims"),
    ("car", "has", "wheels"),
    ("bus", "has", "wheels"),
    ("apple", "is", "fruit"),
    ("banana", "is", "fruit"),
)


def relation_pairs(dataset: str = "tiny_semantics", noise_pairs: int = 0, seed: int = 0) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    if str(dataset) != "tiny_semantics":
        raise ValueError("V1 only ships the tiny_semantics dataset.")
    positive: set[tuple[str, str]] = set()
    tokens: set[str] = set()
    for entry in TINY_SEMANTICS:
        if len(entry) == 2:
            left, right = entry
            tokens.update([left, right])
            positive.add((left, right))
            positive.add((right, left))
            continue
        left, relation, right = entry
        tokens.update([left, relation, right])
        positive.add((left, right))
        positive.add((right, left))
        positive.add((left, relation))
        positive.add((relation, right))

    token_list = sorted(tokens)
    negative: set[tuple[str, str]] = set()
    rng = random.Random(int(seed))
    all_pairs = [(a, b) for a, b in itertools.permutations(token_list, 2) if (a, b) not in positive]
    rng.shuffle(all_pairs)
    for pair in all_pairs[: max(0, int(noise_pairs))]:
        positive.add(pair)
        negative.add(pair)
    return token_list, sorted(positive), sorted(negative)
