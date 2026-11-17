"""Paper-ready table generation from evaluation JSON files.

Reads `evaluation.schema.json` results (each carrying a `meta` provenance block),
aggregates over seeds, and renders Markdown + LaTeX (booktabs) tables.
"""
