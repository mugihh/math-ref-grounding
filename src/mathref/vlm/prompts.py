"""Versioned prompt templates for frozen-VLM candidate ranking."""

import re
from typing import Any


DEFAULT_PROMPT_VERSION = "vlm-ranking-v4"
LATEX_PROMPT_VERSION = "vlm-ranking-latex-v4"

# Default prompt version per ablation condition.
CONDITION_PROMPTS = {
    "vlm": DEFAULT_PROMPT_VERSION,
    "vlm+latex": LATEX_PROMPT_VERSION,
}

_PROMPT_TEMPLATES = {
    "vlm-ranking-v1": """You are given a mathematics keyframe with candidate boxes labeled {candidate_labels}.

Transcript context:
{caption}

Target mention:
{mention}

Rank every candidate label from most likely to least likely to be the visual referent of the target mention.
Return JSON only, using this exact shape:
{{"ranking": ["<candidate_label>", "..."]}}
Include every candidate label exactly once.""",
    "vlm-ranking-v2": """You are given a mathematics keyframe with exactly {candidate_count} candidate boxes labeled {candidate_labels}.

Transcript context:
{caption}

Target mention:
{mention}

Rank all {candidate_count} candidate labels from most likely to least likely to be the visual referent of the target mention.
Before answering, check that your ranking contains exactly {candidate_count} labels, that every listed candidate label occurs exactly once, and that there are no other labels.
Return JSON only, using this exact shape:
{{"ranking": ["<candidate_label>", "..."]}}""",
    "vlm-ranking-v3": """You are given a mathematics keyframe with exactly {candidate_count} candidate boxes labeled {candidate_labels}.

Transcript context:
{caption}

Target mention:
{mention}

Rank all {candidate_count} candidate labels from most likely to least likely to be the visual referent of the target mention.
Before answering, check that your ranking contains exactly {candidate_count} labels, that every listed candidate label occurs exactly once, and that there are no other labels.
Return JSON only, using this exact shape:
{{"ranking": ["<candidate_label>", "..."]}}""",
    "vlm-ranking-latex-v1": """You are given a mathematics keyframe with candidate boxes labeled {candidate_labels}.

Recognized LaTeX per candidate (handwritten, may be noisy or missing):
{candidate_latex_block}

Transcript context:
{caption}

Target mention:
{mention}

Rank every candidate label from most likely to least likely to be the visual referent of the target mention.
Return JSON only, using this exact shape:
{{"ranking": ["<candidate_label>", "..."]}}
Include every candidate label exactly once.""",
    "vlm-ranking-latex-v2": """You are given a mathematics keyframe with exactly {candidate_count} candidate boxes labeled {candidate_labels}.

Recognized LaTeX per candidate (handwritten, may be noisy or missing):
{candidate_latex_block}

Transcript context:
{caption}

Target mention:
{mention}

Rank all {candidate_count} candidate labels from most likely to least likely to be the visual referent of the target mention.
Before answering, check that your ranking contains exactly {candidate_count} labels, that every listed candidate label occurs exactly once, and that there are no other labels.
Return JSON only, using this exact shape:
{{"ranking": ["<candidate_label>", "..."]}}""",
    "vlm-ranking-latex-v3": """You are given a mathematics keyframe with exactly {candidate_count} candidate boxes labeled {candidate_labels}.

Recognized LaTeX per candidate (handwritten, may be noisy or missing):
{candidate_latex_block}

Transcript context:
{caption}

Target mention:
{mention}

Rank all {candidate_count} candidate labels from most likely to least likely to be the visual referent of the target mention.
Before answering, check that your ranking contains exactly {candidate_count} labels, that every listed candidate label occurs exactly once, and that there are no other labels.
Return JSON only, using this exact shape:
{{"ranking": ["<candidate_label>", "..."]}}""",
}

_TAG_RE = re.compile(r"<<<(.*?)>>>", re.DOTALL)
_ANNOTATION_SUFFIX_RE_V3 = re.compile(r"(?<=\w)_[0-9]{2}\b")
_ANNOTATION_SUFFIX_RE_V4 = re.compile(r"(?<=\S)_[0-9]{2}\b")
_PROMPT_TEMPLATES["vlm-ranking-v4"] = _PROMPT_TEMPLATES["vlm-ranking-v3"]
_PROMPT_TEMPLATES["vlm-ranking-latex-v4"] = _PROMPT_TEMPLATES["vlm-ranking-latex-v3"]
_SANITIZED_PROMPT_VERSIONS = {
    "vlm-ranking-v3",
    "vlm-ranking-latex-v3",
    "vlm-ranking-v4",
    "vlm-ranking-latex-v4",
}


def strip_annotation_suffixes(text: str, prompt_version: str = DEFAULT_PROMPT_VERSION) -> str:
    """Remove repeated-mention bookkeeping suffixes such as ``_00``."""
    pattern = (
        _ANNOTATION_SUFFIX_RE_V4
        if prompt_version in {"vlm-ranking-v4", "vlm-ranking-latex-v4"}
        else _ANNOTATION_SUFFIX_RE_V3
    )
    return pattern.sub("", text)


def sanitize_caption_mentions(text: str, prompt_version: str = DEFAULT_PROMPT_VERSION) -> str:
    """Strip bookkeeping suffixes only inside marked transcript mentions."""
    return _TAG_RE.sub(
        lambda match: (
            f"<<<{strip_annotation_suffixes(match.group(1), prompt_version)}>>>"
        ),
        text,
    )


def format_candidate_latex_block(candidate_labels: list[str], candidate_latex: list[str | None]) -> str:
    """One line per candidate; missing recognition is shown, never dropped."""
    lines = []
    for label, latex in zip(candidate_labels, candidate_latex, strict=True):
        lines.append(f"- {label}: {latex if latex else '(no LaTeX available)'}")
    return "\n".join(lines)


def render_prompt(
    example: dict[str, Any],
    candidate_labels: list[str],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    candidate_latex: list[str | None] | None = None,
) -> str:
    """Render a registered prompt without exposing gold-answer fields."""
    try:
        template = _PROMPT_TEMPLATES[prompt_version]
    except KeyError as error:
        supported = ", ".join(sorted(_PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt version {prompt_version!r}; supported: {supported}"
        ) from error
    if not candidate_labels:
        raise ValueError("At least one candidate label is required.")
    if "{candidate_latex_block}" in template and candidate_latex is None:
        raise ValueError(f"Prompt {prompt_version!r} requires candidate_latex.")
    block = (
        format_candidate_latex_block(candidate_labels, candidate_latex)
        if candidate_latex is not None
        else ""
    )
    caption = example["caption"]["text"]
    mention = example["mention"]["text"]
    if prompt_version in _SANITIZED_PROMPT_VERSIONS:
        caption = sanitize_caption_mentions(caption, prompt_version)
        mention = strip_annotation_suffixes(mention, prompt_version)
    return template.format(
        candidate_labels=", ".join(candidate_labels),
        candidate_count=len(candidate_labels),
        candidate_latex_block=block,
        caption=caption,
        mention=mention,
    )
