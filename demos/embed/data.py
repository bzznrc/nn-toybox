"""Tiny hand-written semantic corpus."""

from __future__ import annotations

import itertools
import random

from core.datasets import TEXT_DATASET_KEYS, dataset_display_names, dataset_key


TINY_SEMANTICS = [
    # animals
    ("cat", "is_a", "animal"),
    ("dog", "is_a", "animal"),
    ("rabbit", "is_a", "animal"),
    ("bear", "is_a", "animal"),
    ("bird", "is_a", "animal"),
    ("eagle", "is_a", "bird"),
    ("fish", "is_a", "animal"),
    ("shark", "is_a", "fish"),
    ("duck", "is_a", "bird"),
    ("turtle", "is_a", "animal"),
    ("cat", "has", "fur"),
    ("dog", "has", "fur"),
    ("rabbit", "has", "fur"),
    ("bear", "has", "fur"),
    ("bird", "has", "wings"),
    ("eagle", "has", "wings"),
    ("duck", "has", "wings"),
    ("fish", "has", "fins"),
    ("shark", "has", "fins"),
    ("turtle", "has", "shell"),
    ("bird", "can", "fly"),
    ("eagle", "can", "fly"),
    ("fish", "can", "swim"),
    ("shark", "can", "swim"),
    ("duck", "can", "swim"),
    ("turtle", "can", "swim"),

    # vehicles
    ("car", "is_a", "vehicle"),
    ("bus", "is_a", "vehicle"),
    ("truck", "is_a", "vehicle"),
    ("bike", "is_a", "vehicle"),
    ("train", "is_a", "vehicle"),
    ("plane", "is_a", "vehicle"),
    ("car", "has", "wheels"),
    ("bus", "has", "wheels"),
    ("truck", "has", "wheels"),
    ("bike", "has", "wheels"),
    ("train", "has", "wheels"),
    ("plane", "has", "wings"),
    ("car", "used_for", "travel"),
    ("bus", "used_for", "travel"),
    ("train", "used_for", "travel"),
    ("plane", "used_for", "travel"),
    ("truck", "used_for", "carry"),
    ("bike", "used_for", "ride"),
    ("car", "moves_on", "road"),
    ("bus", "moves_on", "road"),
    ("truck", "moves_on", "road"),
    ("bike", "moves_on", "road"),
    ("train", "moves_on", "rail"),
    ("plane", "moves_in", "sky"),

    # food
    ("apple", "is_a", "fruit"),
    ("banana", "is_a", "fruit"),
    ("orange", "is_a", "fruit"),
    ("grape", "is_a", "fruit"),
    ("carrot", "is_a", "vegetable"),
    ("potato", "is_a", "vegetable"),
    ("lettuce", "is_a", "vegetable"),
    ("apple", "has", "seeds"),
    ("orange", "has", "seeds"),
    ("grape", "has", "seeds"),
    ("banana", "has", "peel"),
    ("orange", "has", "peel"),
    ("carrot", "has", "root"),
    ("potato", "has", "root"),
    ("apple", "tastes", "sweet"),
    ("banana", "tastes", "sweet"),
    ("orange", "tastes", "sweet"),
    ("grape", "tastes", "sweet"),
    ("carrot", "tastes", "sweet"),

    # tools / objects
    ("hammer", "is_a", "tool"),
    ("saw", "is_a", "tool"),
    ("screwdriver", "is_a", "tool"),
    ("wrench", "is_a", "tool"),
    ("spoon", "is_a", "utensil"),
    ("fork", "is_a", "utensil"),
    ("hammer", "used_for", "build"),
    ("saw", "used_for", "cut"),
    ("screwdriver", "used_for", "repair"),
    ("wrench", "used_for", "repair"),
    ("spoon", "used_for", "eat"),
    ("fork", "used_for", "eat"),
    ("hammer", "made_of", "metal"),
    ("saw", "made_of", "metal"),
    ("screwdriver", "made_of", "metal"),
    ("wrench", "made_of", "metal"),
    ("spoon", "made_of", "metal"),
    ("fork", "made_of", "metal"),
    
]

BIG_TINY_SEMANTICS = [
    # animals: category
    ("cat", "is_a", "animal"),
    ("dog", "is_a", "animal"),
    ("rabbit", "is_a", "animal"),
    ("horse", "is_a", "animal"),
    ("cow", "is_a", "animal"),
    ("sheep", "is_a", "animal"),
    ("bear", "is_a", "animal"),
    ("lion", "is_a", "animal"),
    ("bird", "is_a", "animal"),
    ("eagle", "is_a", "animal"),
    ("duck", "is_a", "animal"),
    ("fish", "is_a", "animal"),
    ("shark", "is_a", "animal"),
    ("turtle", "is_a", "animal"),
    ("frog", "is_a", "animal"),
    ("bee", "is_a", "animal"),

    # animals: traits
    ("cat", "has", "fur"),
    ("dog", "has", "fur"),
    ("rabbit", "has", "fur"),
    ("horse", "has", "fur"),
    ("cow", "has", "fur"),
    ("sheep", "has", "wool"),
    ("bear", "has", "fur"),
    ("lion", "has", "fur"),
    ("bird", "has", "wings"),
    ("eagle", "has", "wings"),
    ("duck", "has", "wings"),
    ("fish", "has", "fins"),
    ("shark", "has", "fins"),
    ("turtle", "has", "shell"),
    ("frog", "has", "legs"),
    ("bee", "has", "wings"),

    # animals: behavior
    ("cat", "can", "climb"),
    ("dog", "can", "run"),
    ("rabbit", "can", "hop"),
    ("horse", "can", "run"),
    ("cow", "can", "graze"),
    ("sheep", "can", "graze"),
    ("bear", "can", "climb"),
    ("lion", "can", "hunt"),
    ("bird", "can", "fly"),
    ("eagle", "can", "fly"),
    ("duck", "can", "swim"),
    ("fish", "can", "swim"),
    ("shark", "can", "swim"),
    ("turtle", "can", "swim"),
    ("frog", "can", "jump"),
    ("bee", "can", "fly"),

    # vehicles: category
    ("car", "is_a", "vehicle"),
    ("bus", "is_a", "vehicle"),
    ("truck", "is_a", "vehicle"),
    ("bike", "is_a", "vehicle"),
    ("scooter", "is_a", "vehicle"),
    ("motorcycle", "is_a", "vehicle"),
    ("train", "is_a", "vehicle"),
    ("tram", "is_a", "vehicle"),
    ("boat", "is_a", "vehicle"),
    ("ship", "is_a", "vehicle"),
    ("plane", "is_a", "vehicle"),
    ("helicopter", "is_a", "vehicle"),
    ("rocket", "is_a", "vehicle"),
    ("tractor", "is_a", "vehicle"),
    ("van", "is_a", "vehicle"),
    ("taxi", "is_a", "vehicle"),

    # vehicles: traits
    ("car", "has", "wheels"),
    ("bus", "has", "wheels"),
    ("truck", "has", "wheels"),
    ("bike", "has", "wheels"),
    ("scooter", "has", "wheels"),
    ("motorcycle", "has", "wheels"),
    ("train", "has", "wheels"),
    ("tram", "has", "wheels"),
    ("boat", "has", "hull"),
    ("ship", "has", "hull"),
    ("plane", "has", "wings"),
    ("helicopter", "has", "rotor"),
    ("rocket", "has", "engine"),
    ("tractor", "has", "wheels"),
    ("van", "has", "wheels"),
    ("taxi", "has", "wheels"),

    # vehicles: movement/use
    ("car", "moves_on", "road"),
    ("bus", "moves_on", "road"),
    ("truck", "moves_on", "road"),
    ("bike", "moves_on", "road"),
    ("scooter", "moves_on", "road"),
    ("motorcycle", "moves_on", "road"),
    ("train", "moves_on", "rail"),
    ("tram", "moves_on", "rail"),
    ("boat", "moves_on", "water"),
    ("ship", "moves_on", "water"),
    ("plane", "moves_in", "sky"),
    ("helicopter", "moves_in", "sky"),
    ("rocket", "moves_in", "space"),
    ("tractor", "moves_on", "field"),
    ("van", "moves_on", "road"),
    ("taxi", "moves_on", "road"),

    # food: category
    ("apple", "is_a", "food"),
    ("banana", "is_a", "food"),
    ("orange", "is_a", "food"),
    ("grape", "is_a", "food"),
    ("strawberry", "is_a", "food"),
    ("carrot", "is_a", "food"),
    ("potato", "is_a", "food"),
    ("lettuce", "is_a", "food"),
    ("bread", "is_a", "food"),
    ("rice", "is_a", "food"),
    ("pasta", "is_a", "food"),
    ("cheese", "is_a", "food"),
    ("milk", "is_a", "food"),
    ("egg", "is_a", "food"),
    ("fish_food", "is_a", "food"),
    ("honey", "is_a", "food"),

    # food: subcategories
    ("apple", "is_a", "fruit"),
    ("banana", "is_a", "fruit"),
    ("orange", "is_a", "fruit"),
    ("grape", "is_a", "fruit"),
    ("strawberry", "is_a", "fruit"),
    ("carrot", "is_a", "vegetable"),
    ("potato", "is_a", "vegetable"),
    ("lettuce", "is_a", "vegetable"),
    ("bread", "is_a", "grain"),
    ("rice", "is_a", "grain"),
    ("pasta", "is_a", "grain"),
    ("cheese", "is_a", "dairy"),
    ("milk", "is_a", "dairy"),
    ("egg", "is_a", "protein"),
    ("fish_food", "is_a", "protein"),
    ("honey", "is_a", "sweet"),

    # food: traits
    ("apple", "tastes", "sweet"),
    ("banana", "tastes", "sweet"),
    ("orange", "tastes", "sweet"),
    ("grape", "tastes", "sweet"),
    ("strawberry", "tastes", "sweet"),
    ("carrot", "tastes", "sweet"),
    ("potato", "tastes", "earthy"),
    ("lettuce", "tastes", "fresh"),
    ("bread", "tastes", "mild"),
    ("rice", "tastes", "mild"),
    ("pasta", "tastes", "mild"),
    ("cheese", "tastes", "savory"),
    ("milk", "tastes", "mild"),
    ("egg", "tastes", "savory"),
    ("fish_food", "tastes", "savory"),
    ("honey", "tastes", "sweet"),

    # tools: category
    ("hammer", "is_a", "tool"),
    ("saw", "is_a", "tool"),
    ("wrench", "is_a", "tool"),
    ("screwdriver", "is_a", "tool"),
    ("drill", "is_a", "tool"),
    ("pliers", "is_a", "tool"),
    ("knife", "is_a", "tool"),
    ("scissors", "is_a", "tool"),
    ("shovel", "is_a", "tool"),
    ("rake", "is_a", "tool"),
    ("brush", "is_a", "tool"),
    ("pen", "is_a", "tool"),
    ("pencil", "is_a", "tool"),
    ("spoon", "is_a", "tool"),
    ("fork", "is_a", "tool"),
    ("needle", "is_a", "tool"),

    # tools: use
    ("hammer", "used_for", "build"),
    ("saw", "used_for", "cut"),
    ("wrench", "used_for", "repair"),
    ("screwdriver", "used_for", "repair"),
    ("drill", "used_for", "build"),
    ("pliers", "used_for", "repair"),
    ("knife", "used_for", "cut"),
    ("scissors", "used_for", "cut"),
    ("shovel", "used_for", "dig"),
    ("rake", "used_for", "gather"),
    ("brush", "used_for", "paint"),
    ("pen", "used_for", "write"),
    ("pencil", "used_for", "write"),
    ("spoon", "used_for", "eat"),
    ("fork", "used_for", "eat"),
    ("needle", "used_for", "sew"),

    # tools: material/shape
    ("hammer", "made_of", "metal"),
    ("saw", "made_of", "metal"),
    ("wrench", "made_of", "metal"),
    ("screwdriver", "made_of", "metal"),
    ("drill", "made_of", "metal"),
    ("pliers", "made_of", "metal"),
    ("knife", "made_of", "metal"),
    ("scissors", "made_of", "metal"),
    ("shovel", "made_of", "metal"),
    ("rake", "made_of", "metal"),
    ("brush", "has", "handle"),
    ("pen", "has", "tip"),
    ("pencil", "has", "tip"),
    ("spoon", "made_of", "metal"),
    ("fork", "made_of", "metal"),
    ("needle", "made_of", "metal"),

    # deliberate cross-category bridges
    ("bird", "has", "wings"),
    ("plane", "has", "wings"),
    ("duck", "can", "swim"),
    ("boat", "moves_on", "water"),
    ("fish", "can", "swim"),
    ("ship", "moves_on", "water"),
    ("horse", "used_for", "travel"),
    ("bike", "used_for", "travel"),
    ("truck", "used_for", "carry"),
    ("shovel", "used_for", "carry"),
    ("knife", "used_for", "cut"),
    ("bread", "can_be", "cut"),
    ("spoon", "used_for", "eat"),
    ("food", "can_be", "eat"),
]

DATASETS = dataset_display_names(TEXT_DATASET_KEYS)

ANIMAL_TOKENS = (
    "animal",
    "cat",
    "dog",
    "rabbit",
    "horse",
    "cow",
    "sheep",
    "bear",
    "lion",
    "bird",
    "eagle",
    "duck",
    "fish",
    "shark",
    "turtle",
    "frog",
    "bee",
)
VEHICLE_TOKENS = (
    "vehicle",
    "car",
    "bus",
    "truck",
    "bike",
    "scooter",
    "motorcycle",
    "train",
    "tram",
    "boat",
    "ship",
    "plane",
    "helicopter",
    "rocket",
    "tractor",
    "van",
    "taxi",
)
FOOD_TOKENS = (
    "food",
    "fruit",
    "vegetable",
    "grain",
    "dairy",
    "protein",
    "sweet",
    "apple",
    "banana",
    "orange",
    "grape",
    "strawberry",
    "carrot",
    "potato",
    "lettuce",
    "bread",
    "rice",
    "pasta",
    "cheese",
    "milk",
    "egg",
    "fish_food",
    "honey",
)
TOOL_TOKENS = (
    "tool",
    "utensil",
    "hammer",
    "saw",
    "wrench",
    "screwdriver",
    "drill",
    "pliers",
    "knife",
    "scissors",
    "shovel",
    "rake",
    "brush",
    "pen",
    "pencil",
    "spoon",
    "fork",
    "needle",
)

TOKEN_GROUPS: dict[str, str] = {
    **dict.fromkeys(ANIMAL_TOKENS, "animals"),
    **dict.fromkeys(VEHICLE_TOKENS, "vehicles"),
    **dict.fromkeys(FOOD_TOKENS, "food"),
    **dict.fromkeys(TOOL_TOKENS, "tools"),
}


def token_group(token: str) -> str | None:
    return TOKEN_GROUPS.get(str(token))


def relation_pairs(dataset: str = "Text - Tiny Semantics", noise_pairs: int = 0, seed: int = 0) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    key = dataset_key(dataset)
    if key == "tiny_semantics":
        entries = TINY_SEMANTICS
    elif key == "big_tiny_semantics":
        entries = BIG_TINY_SEMANTICS
    else:
        valid = ", ".join(DATASETS)
        raise ValueError(f"Unknown embed dataset '{dataset}'. Valid: {valid}")
    positive: set[tuple[str, str]] = set()
    tokens: set[str] = set()
    for entry in entries:
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
