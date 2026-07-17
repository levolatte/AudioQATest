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

    Args:
        question: The question text.
        choices: List of option strings.
        label_only: If True, prompt the model to output ONLY the option label
                    (e.g. "A") without any explanation. Choices are presented
                    with letter prefixes (A/B/C/D).
                    If False, prompt the model to output the full answer with
                    explanation inside <answer> tags.
    """
    labels = ["A", "B", "C", "D", "E", "F"][:len(choices)]

    if label_only:
        # Label-only mode: letter-prefixed options, output ONLY the letter
        labeled_options = [f"{labels[i]}. {choices[i]}" for i in range(len(choices))]
        return (
            f"{question}\n"
            + "\n".join(labeled_options) +
            f"\n\nOutput ONLY the letter of the correct choice "
            f"(e.g. A, B, C, or D). Do NOT include any explanation or the "
            f"option text.\n\nAnswer:"
        )

    # Standard mode: plain option list, output with <answer> tags
    options_str = [str(c) for c in choices]
    labeled_options = [f"{labels[i]}. {choices[i]}" for i in range(len(choices))]
    return (
        f"{question}\n"
        + "\n".join(labeled_options) +
        f"\n\nPlease choose the answer from the following options: "
        f"{options_str}. "
        f"Output the final answer in <answer> </answer>."
    )


def clean_answer(pred, choices: list[str]) -> str:
    """Extract the chosen answer from raw model output.

    Multi-level fallback strategy (from infer.py):
    1. Parse <answer>...</answer> tags
    2. Clean special tokens
    3. Exact match against option text
    4. Letter fallback (A/B/C/D)
    5. Contains match (option text in output)
    6. Reverse contains match (output in option text)
    7. Default to first choice
    """
    if pred is None or len(choices) == 0:
        return choices[0] if choices else ""

    raw = str(pred).strip()

    # Priority: parse <answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", raw, flags=re.I | re.S)
    if m:
        raw = m.group(1).strip()

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
