"""Synthetic subject-verb agreement data for the attend demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SUBJECTS = (
    ("dog", "singular"),
    ("dogs", "plural"),
    ("cat", "singular"),
    ("cats", "plural"),
    ("bird", "singular"),
    ("birds", "plural"),
    ("car", "singular"),
    ("cars", "plural"),
    ("fox", "singular"),
    ("foxes", "plural"),
    ("duck", "singular"),
    ("ducks", "plural"),
)
DISTRACTORS = (
    ("dog", "singular"),
    ("dogs", "plural"),
    ("cat", "singular"),
    ("cats", "plural"),
    ("bird", "singular"),
    ("birds", "plural"),
    ("car", "singular"),
    ("cars", "plural"),
    ("tree", "singular"),
    ("trees", "plural"),
    ("boat", "singular"),
    ("boats", "plural"),
)
PREPOSITIONS = ("near", "behind", "beside", "above")
ADJECTIVES = ("noisy", "sleepy", "small", "bright", "loud", "quiet")
TEMPLATES = (
    ("the", "{S}", "{P}", "the", "{D}", "[MASK]", "{A}"),
    ("{P}", "the", "{D}", ",", "the", "{S}", "[MASK]", "{A}"),
    ("the", "{A}", "{S}", "{P}", "the", "{D}", "[MASK]", "here"),
    ("today", "the", "{S}", "{P}", "the", "{D}", "[MASK]", "{A}"),
)
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
LABELS = ("is", "are")
MAX_SENTENCE_LENGTH = max(len(template) for template in TEMPLATES)


@dataclass(frozen=True)
class AgreementExample:
    tokens: tuple[str, ...]
    target: int
    subject_index: int
    distractor_index: int
    mask_index: int
    subject_number: str
    distractor_number: str
    template_id: int
    trap: bool

    @property
    def target_label(self) -> str:
        return LABELS[int(self.target)]

    @property
    def subject(self) -> str:
        return self.tokens[int(self.subject_index)]

    @property
    def distractor(self) -> str:
        return self.tokens[int(self.distractor_index)]


@dataclass(frozen=True)
class AgreementBatch:
    token_ids: np.ndarray
    token_mask: np.ndarray
    targets: np.ndarray
    subject_indices: np.ndarray
    distractor_indices: np.ndarray
    mask_indices: np.ndarray
    lengths: np.ndarray
    template_ids: np.ndarray
    traps: np.ndarray
    subject_numbers: tuple[str, ...]
    distractor_numbers: tuple[str, ...]


def build_vocab() -> tuple[str, ...]:
    words = {PAD_TOKEN, MASK_TOKEN, *PREPOSITIONS, *ADJECTIVES}
    for template in TEMPLATES:
        words.update(token for token in template if not token.startswith("{"))
    words.update(word for word, _number in SUBJECTS)
    words.update(word for word, _number in DISTRACTORS)
    return tuple([PAD_TOKEN, *sorted(word for word in words if word != PAD_TOKEN)])


VOCAB = build_vocab()
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}


def target_for_number(number: str) -> int:
    return 0 if str(number) == "singular" else 1


def _render_template(
    template: tuple[str, ...],
    *,
    subject_word: str,
    distractor_word: str,
    prep: str,
    adjective: str,
) -> tuple[tuple[str, ...], int, int, int]:
    tokens: list[str] = []
    subject_index = -1
    distractor_index = -1
    mask_index = -1
    for raw in template:
        if raw == "{S}":
            subject_index = len(tokens)
            tokens.append(str(subject_word))
        elif raw == "{D}":
            distractor_index = len(tokens)
            tokens.append(str(distractor_word))
        elif raw == "{P}":
            tokens.append(str(prep))
        elif raw == "{A}":
            tokens.append(str(adjective))
        elif raw == MASK_TOKEN:
            mask_index = len(tokens)
            tokens.append(MASK_TOKEN)
        else:
            tokens.append(str(raw))
    return tuple(tokens), subject_index, distractor_index, mask_index


def make_example(
    subject: tuple[str, str],
    distractor: tuple[str, str],
    prep: str,
    adjective: str,
    *,
    template_id: int = 0,
) -> AgreementExample:
    subject_word, subject_number = subject
    distractor_word, distractor_number = distractor
    template_index = int(template_id) % len(TEMPLATES)
    tokens, subject_index, distractor_index, mask_index = _render_template(
        TEMPLATES[template_index],
        subject_word=subject_word,
        distractor_word=distractor_word,
        prep=str(prep),
        adjective=str(adjective),
    )
    return AgreementExample(
        tokens=tokens,
        target=target_for_number(subject_number),
        subject_index=subject_index,
        distractor_index=distractor_index,
        mask_index=mask_index,
        subject_number=subject_number,
        distractor_number=distractor_number,
        template_id=template_index + 1,
        trap=bool(subject_number != distractor_number),
    )


def all_examples() -> tuple[AgreementExample, ...]:
    return tuple(
        make_example(subject, distractor, prep, adjective, template_id=template_idx)
        for template_idx in range(len(TEMPLATES))
        for subject in SUBJECTS
        for distractor in DISTRACTORS
        for prep in PREPOSITIONS
        for adjective in ADJECTIVES
    )


DISPLAY_EXAMPLES = (
    make_example(("dogs", "plural"), ("cat", "singular"), "near", "noisy", template_id=0),
    make_example(("dog", "singular"), ("cats", "plural"), "near", "noisy", template_id=1),
    make_example(("dogs", "plural"), ("cat", "singular"), "beside", "noisy", template_id=2),
    make_example(("dog", "singular"), ("cats", "plural"), "behind", "loud", template_id=3),
    make_example(("fox", "singular"), ("boats", "plural"), "beside", "quiet", template_id=1),
    make_example(("ducks", "plural"), ("tree", "singular"), "above", "bright", template_id=3),
)


def has_trap_cases(examples: tuple[AgreementExample, ...]) -> bool:
    has_singular_subject_plural_distractor = any(
        example.subject_number == "singular" and example.distractor_number == "plural" for example in examples
    )
    has_plural_subject_singular_distractor = any(
        example.subject_number == "plural" and example.distractor_number == "singular" for example in examples
    )
    return bool(has_singular_subject_plural_distractor and has_plural_subject_singular_distractor)


def _choice_with_number(
    rng: np.random.Generator,
    table: tuple[tuple[str, str], ...],
    number: str,
) -> tuple[str, str]:
    candidates = tuple(item for item in table if item[1] == str(number))
    return candidates[int(rng.integers(0, len(candidates)))]


def random_example(rng: np.random.Generator, *, trap_rate: float = 0.75) -> AgreementExample:
    subject = SUBJECTS[int(rng.integers(0, len(SUBJECTS)))]
    if rng.random() < max(0.0, min(1.0, float(trap_rate))):
        distractor_number = "plural" if subject[1] == "singular" else "singular"
        distractor = _choice_with_number(rng, DISTRACTORS, distractor_number)
    else:
        distractor = DISTRACTORS[int(rng.integers(0, len(DISTRACTORS)))]
    prep = PREPOSITIONS[int(rng.integers(0, len(PREPOSITIONS)))]
    adjective = ADJECTIVES[int(rng.integers(0, len(ADJECTIVES)))]
    template_id = int(rng.integers(0, len(TEMPLATES)))
    return make_example(subject, distractor, prep, adjective, template_id=template_id)


def trap_example(rng: np.random.Generator, *, subject_number: str) -> AgreementExample:
    distractor_number = "plural" if str(subject_number) == "singular" else "singular"
    subject = _choice_with_number(rng, SUBJECTS, str(subject_number))
    distractor = _choice_with_number(rng, DISTRACTORS, distractor_number)
    prep = PREPOSITIONS[int(rng.integers(0, len(PREPOSITIONS)))]
    adjective = ADJECTIVES[int(rng.integers(0, len(ADJECTIVES)))]
    template_id = int(rng.integers(0, len(TEMPLATES)))
    return make_example(subject, distractor, prep, adjective, template_id=template_id)


def encode_tokens(tokens: tuple[str, ...]) -> np.ndarray:
    ids = np.full((MAX_SENTENCE_LENGTH,), TOKEN_TO_ID[PAD_TOKEN], dtype=np.int64)
    encoded = [TOKEN_TO_ID[token] for token in tokens]
    ids[: len(encoded)] = np.asarray(encoded, dtype=np.int64)
    return ids


def batch_from_examples(examples: tuple[AgreementExample, ...]) -> AgreementBatch:
    lengths = np.asarray([len(example.tokens) for example in examples], dtype=np.int64)
    token_mask = np.zeros((len(examples), MAX_SENTENCE_LENGTH), dtype=bool)
    for idx, length in enumerate(lengths):
        token_mask[idx, : int(length)] = True
    return AgreementBatch(
        token_ids=np.stack([encode_tokens(example.tokens) for example in examples], axis=0),
        token_mask=token_mask,
        targets=np.asarray([example.target for example in examples], dtype=np.int64),
        subject_indices=np.asarray([example.subject_index for example in examples], dtype=np.int64),
        distractor_indices=np.asarray([example.distractor_index for example in examples], dtype=np.int64),
        mask_indices=np.asarray([example.mask_index for example in examples], dtype=np.int64),
        lengths=lengths,
        template_ids=np.asarray([example.template_id for example in examples], dtype=np.int64),
        traps=np.asarray([example.trap for example in examples], dtype=bool),
        subject_numbers=tuple(example.subject_number for example in examples),
        distractor_numbers=tuple(example.distractor_number for example in examples),
    )


def make_batch(
    rng: np.random.Generator,
    *,
    batch_size: int,
    trap_rate: float = 0.75,
    force_traps: bool = True,
) -> AgreementBatch:
    size = max(1, int(batch_size))
    examples: list[AgreementExample] = []
    if bool(force_traps) and size >= 2:
        examples.append(trap_example(rng, subject_number="singular"))
        examples.append(trap_example(rng, subject_number="plural"))
    while len(examples) < size:
        examples.append(random_example(rng, trap_rate=float(trap_rate)))
    return batch_from_examples(tuple(examples[:size]))
