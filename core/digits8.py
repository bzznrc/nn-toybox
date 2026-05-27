"""Zip-backed loader for the bundled 8x8 digit mini dataset."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from core.config import PROJECT_ROOT


DIGIT_LABELS: tuple[int, ...] = tuple(range(10))
DIGITS8_ZIP_NAME = "digits8_mini_8x8.zip"
PACKAGE_ASSET_MODULE = "assets"
DEFAULT_DIGITS8_ZIP = PROJECT_ROOT / "assets" / DIGITS8_ZIP_NAME
VALID_SPLITS = ("train", "inference")


@dataclass(frozen=True)
class Digits8Split:
    images: np.ndarray
    labels: np.ndarray
    names: tuple[str, ...]


class Digits8Dataset:
    """Small array-backed dataset compatible with torch-style indexing."""

    def __init__(self, split: str = "train", *, channel_first: bool = True, zip_path: str | Path | None = None) -> None:
        loaded = load_digits8_split(split=split, channel_first=channel_first, zip_path=zip_path)
        self.images = loaded.images
        self.labels = loaded.labels
        self.names = loaded.names
        self.split = split

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        idx = int(index)
        return self.images[idx], int(self.labels[idx])


class Digits8Browser:
    """Shared deterministic browsing semantics for digit demo viewers.

    Up/down move through variations of the current digit. Left/right move to the
    previous/next digit while keeping the same variation slot when possible.
    """

    def __init__(self, labels: np.ndarray, *, selected_index: int = 0) -> None:
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if self.labels.size == 0:
            raise ValueError("Digits8Browser needs at least one label")
        self.indices_by_digit: dict[int, np.ndarray] = {
            int(label): np.flatnonzero(self.labels == int(label)).astype(np.int64)
            for label in sorted(int(item) for item in np.unique(self.labels))
        }
        self.digits = tuple(self.indices_by_digit.keys())
        self.selected_index = int(selected_index) % int(self.labels.size)

    @property
    def selected_digit(self) -> int:
        return int(self.labels[int(self.selected_index)])

    @property
    def variation_index(self) -> int:
        indices = self.indices_by_digit[self.selected_digit]
        matches = np.flatnonzero(indices == int(self.selected_index))
        return int(matches[0]) if matches.size else 0

    def cycle_variation(self, delta: int = 1) -> int:
        indices = self.indices_by_digit[self.selected_digit]
        pos = self.variation_index
        self.selected_index = int(indices[(pos + int(delta)) % len(indices)])
        return int(self.selected_index)

    def cycle_digit(self, delta: int = 1) -> int:
        current_digit = self.selected_digit
        digit_pos = self.digits.index(current_digit)
        next_digit = self.digits[(digit_pos + int(delta)) % len(self.digits)]
        variation = self.variation_index
        indices = self.indices_by_digit[next_digit]
        self.selected_index = int(indices[variation % len(indices)])
        return int(self.selected_index)

    def random(self, rng: np.random.Generator, *, digit: int | None = None) -> int:
        if digit is None:
            self.selected_index = int(rng.integers(0, int(self.labels.size)))
            return int(self.selected_index)
        key = int(digit)
        indices = self.indices_by_digit.get(key)
        if indices is None or len(indices) == 0:
            raise ValueError(f"Digit {key} is not available in this split")
        self.selected_index = int(rng.choice(indices))
        return int(self.selected_index)


def _normalize_split(split: str) -> str:
    key = str(split).strip().lower()
    if key not in VALID_SPLITS:
        valid = ", ".join(VALID_SPLITS)
        raise ValueError(f"Unknown digits8 split '{split}'. Valid: {valid}")
    return key


def _resolve_zip_path(zip_path: str | Path | None = None) -> Path:
    path = Path(zip_path).expanduser() if zip_path is not None else DEFAULT_DIGITS8_ZIP
    target = path.resolve()
    if not target.exists():
        raise FileNotFoundError(f"Missing bundled digits8 asset at {target}")
    return target


def packaged_digits8_asset():
    """Return the importlib resource for the packaged digits zip."""

    return files(PACKAGE_ASSET_MODULE).joinpath(DIGITS8_ZIP_NAME)


@lru_cache(maxsize=4)
def _load_split_cached(split: str, zip_path_text: str) -> Digits8Split:
    split_key = _normalize_split(split)
    zip_path = Path(zip_path_text)
    prefix = f"digits8_mini/{split_key}/"
    images: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith(prefix) and name.lower().endswith(".png") and not name.endswith("/")
        ]
        members.sort(key=lambda name: (int(name.split("/")[-2]), name))
        for member in members:
            label = int(member.split("/")[-2])
            with archive.open(member) as file_obj:
                with Image.open(BytesIO(file_obj.read())) as image:
                    arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            if arr.shape != (8, 8):
                raise ValueError(f"Expected 8x8 PNG in {member}, got {arr.shape}")
            images.append(arr)
            labels.append(label)
            names.append(member)
    if not images:
        raise ValueError(f"No PNG files found for digits8 split '{split_key}' in {zip_path}")
    return Digits8Split(
        images=np.stack(images).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        names=tuple(names),
    )


def load_digits8_split(
    split: str = "train",
    *,
    channel_first: bool = True,
    zip_path: str | Path | None = None,
) -> Digits8Split:
    """Load one bundled split into memory without extracting the zip.

    Images are normalized to ``[0, 1]``. With ``channel_first=True`` the shape is
    ``[N, 1, 8, 8]``; otherwise it is ``[N, 8, 8]``.
    """

    split_key = _normalize_split(split)
    if zip_path is not None:
        target = _resolve_zip_path(zip_path)
        loaded = _load_split_cached(split_key, str(target))
    else:
        try:
            resource = packaged_digits8_asset()
            if not resource.is_file():
                raise FileNotFoundError(str(resource))
            with as_file(resource) as target:
                loaded = _load_split_cached(split_key, str(target.resolve()))
        except (FileNotFoundError, ModuleNotFoundError):
            target = _resolve_zip_path(DEFAULT_DIGITS8_ZIP)
            loaded = _load_split_cached(split_key, str(target))
    images = loaded.images
    if channel_first:
        images = images[:, None, :, :]
    return Digits8Split(
        images=images.copy(),
        labels=loaded.labels.copy(),
        names=loaded.names,
    )
