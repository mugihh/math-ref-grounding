# Reviewed candidate LaTeX

`candidate_latex.jsonl` is the frozen, human-reviewed candidate-level LaTeX
artifact used to construct the LaTeX-augmented benchmark examples.

- Frozen: 2026-07-24
- Candidate records: 1,348
- Reviewed non-null LaTeX records: 1,291
- Intentional null records: 57 (`figure_part`: 49; `figure`: 8)
- SHA-256: `7d398084190680772c48b15dfc502acfcf97015960e241066ebffff5755707aa`

Each row is keyed by `image_path` and `visual_id`. The null records correspond
only to non-mathematical figure or figure-part candidates. The file is the
reviewed source artifact; generated `examples_latex.jsonl` files can be rebuilt
from it and the benchmark examples.
