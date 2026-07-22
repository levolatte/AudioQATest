"""Shared text processing utilities extracted from infer.py.

These are used by all model adapters for:
- Building the multiple-choice question prompt
- Normalizing text for answer comparison
- Extracting the chosen answer from raw model output
"""

import re


def normalize_text(x: str) -> str:
    """Normalize text: lowercase, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", str(x).strip().lower()) if x is not None else ""


def build_question_prompt(question: str, choices: list[str],
                          label_only: bool = False) -> str:
    """Build a multiple-choice prompt from question and choices.

    All models use this same prompt format. Choices are presented with letter
    prefixes (A/B/C/D).

    Args:
        question: The question text.
        choices: List of option strings.
        label_only: If True, prompt the model to output ONLY the option letter
                    (e.g. "A") without any explanation.
                    If False, prompt the model to output the letter AND the
                    option text together (e.g. "A. Man").
    """
    labels = ["A", "B", "C", "D", "E", "F"][:len(choices)]
    labeled_options = [f"{labels[i]}. {choices[i]}" for i in range(len(choices))]

    if label_only:
        return (
            "Listen to the audio carefully and answer the question.\n\n"
            f"{question}\n\n"
            + "\n".join(labeled_options) +
            "\n\nChoose the correct answer. Reply with the letter only."
        )

    return (
        "Listen to the audio carefully and answer the question.\n\n"
        f"{question}\n\n"
        + "\n".join(labeled_options) +
        "\n\nChoose the correct answer. Reply with the letter and the corresponding option text."
    )


def clean_answer(pred, choices: list[str]) -> str:
    """Extract the chosen answer from raw model output.

    Multi-level fallback strategy:
    1. Clean special tokens
    2. Exact match against option text
    3. Letter fallback (A/B/C/D)
    4. Contains match (option text in output)
    5. Reverse contains match (output in option text)
    6. Default to first choice
    """
    if pred is None or len(choices) == 0:
        return choices[0] if choices else ""

    raw = str(pred).strip()

    raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    raw = raw.strip("\"'""").strip()

    raw_norm = normalize_text(raw)

    # Exact option text match
    for c in choices:
        if raw_norm == normalize_text(c):
            return c

    # Letter fallback
    upper = raw.upper().strip()
    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    if upper in letter_map and letter_map[upper] < len(choices):
        return choices[letter_map[upper]]

    m = re.search(r"\b([ABCDEF])\b", upper)
    if m:
        idx = letter_map[m.group(1)]
        if idx < len(choices):
            return choices[idx]

    # Contains option text
    sorted_choices = sorted(choices, key=lambda x: len(str(x)), reverse=True)

    for c in sorted_choices:
        c_norm = normalize_text(c)
        if c_norm and c_norm in raw_norm:
            return c

    for c in sorted_choices:
        c_norm = normalize_text(c)
        if raw_norm and raw_norm in c_norm:
            return c

    return choices[0]


def check_strict_match(raw: str, ground_truth: str, choices: list[str]) -> bool:
    """Check if raw output is in strict ``letter. text`` format AND correct.

    A strict match requires:
    1. The output begins with a valid letter (A/B/C/D/...), period, and space
    2. The text after matches the option text at that letter index
    3. The resolved choice equals the ground truth

    Returns True only when all three conditions are met.
    """
    if not raw or not choices:
        return False

    labels = ["A", "B", "C", "D", "E", "F"]
    m = re.match(r"^([A-F])\.\s+(.+)", str(raw).strip())
    if not m:
        return False

    letter = m.group(1)
    text = m.group(2).strip()

    idx = labels.index(letter) if letter in labels else -1
    if idx < 0 or idx >= len(choices):
        return False

    if normalize_text(text) != normalize_text(choices[idx]):
        return False

    return choices[idx] == ground_truth
