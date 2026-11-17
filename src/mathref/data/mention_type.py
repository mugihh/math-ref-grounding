"""Deterministic provisional mention-type classification."""

import re


MENTION_TYPES = {"implicit", "explicit", "unknown"}

_EN_IMPLICIT = re.compile(
    r"^(?:it|this|that|these|those|here|there)(?:\b|$)",
    flags=re.IGNORECASE,
)
_JP_IMPLICIT_PREFIXES = (
    "これ",
    "それ",
    "あれ",
    "こいつ",
    "ここ",
    "そこ",
    "あそこ",
    "この",
    "その",
    "あの",
    "こう",
    "そう",
    "ああ",
    "こんな",
    "そんな",
    "あんな",
)


def classify_mention_type(text: str, language: str) -> str:
    """Classify a mention as implicit, explicit, or unknown.

    The rules are intentionally conservative and provisional. ``unknown`` is
    reserved for missing or unusable text rather than ordinary ambiguity.
    """
    normalized = re.sub(r"_\d{2}$", "", text.strip())
    if not normalized:
        return "unknown"
    if language.lower().startswith("en"):
        return "implicit" if _EN_IMPLICIT.match(normalized) else "explicit"
    if language.lower().startswith(("jp", "ja")):
        return (
            "implicit"
            if normalized.startswith(_JP_IMPLICIT_PREFIXES)
            else "explicit"
        )
    return "unknown"
