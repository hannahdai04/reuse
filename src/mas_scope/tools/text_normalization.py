"""Answer normalization helpers."""

from __future__ import annotations

import re
import string


def normalize_answer(text) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())
