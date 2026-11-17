"""Candidate LaTeX metadata pipeline.

Stage A (no GPU): export candidate crops, build a human-review spreadsheet,
and merge reviewed LaTeX back into ranking examples via a stable composite key.
Stage B (GPU): a separate runner fills the OCR column with a frozen model.
"""
