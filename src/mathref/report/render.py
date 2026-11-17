"""Generic table model and Markdown / LaTeX renderers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Num:
    """A numeric cell holding raw metric values in [0, 1]; formatted at render time."""

    mean: float
    std: float | None = None

    def _parts(self, percent: bool, decimals: int | None) -> tuple[str, str | None]:
        scale = 100 if percent else 1
        places = decimals if decimals is not None else (1 if percent else 3)
        value = f"{self.mean * scale:.{places}f}"
        if self.std is None:
            return value, None
        return value, f"{self.std * scale:.{places}f}"

    def markdown(self, percent: bool = False, decimals: int | None = None) -> str:
        value, std = self._parts(percent, decimals)
        return value if std is None else f"{value} ± {std}"

    def latex(self, percent: bool = False, decimals: int | None = None) -> str:
        value, std = self._parts(percent, decimals)
        return value if std is None else f"{value} $\\pm$ {std}"


Cell = str | Num
MISSING = "—"


@dataclass
class Table:
    title: str
    headers: list[str]
    rows: list[list[Cell]]
    caption: str = ""
    label: str = ""
    align: str = ""  # LaTeX column spec; derived if empty
    field_names: list[str] = field(default_factory=list)  # optional, for reference


def _cell_md(cell: Cell, percent: bool, decimals: int | None) -> str:
    return cell.markdown(percent, decimals) if isinstance(cell, Num) else str(cell)


def _escape_tex(text: str) -> str:
    return (
        text.replace("&", "\\&")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("±", "$\\pm$")
    )


def _cell_tex(cell: Cell, percent: bool, decimals: int | None) -> str:
    if isinstance(cell, Num):
        return cell.latex(percent, decimals)
    return _escape_tex(str(cell))


def _caption(text: str, percent: bool) -> str:
    if percent and text:
        return f"{text} Values are percentages."
    return text


def render_markdown(table: Table, percent: bool = False, decimals: int | None = None) -> str:
    lines = []
    if table.title:
        lines.append(f"### {table.title}")
        lines.append("")
    lines.append("| " + " | ".join(table.headers) + " |")
    lines.append("|" + "|".join(["---"] * len(table.headers)) + "|")
    for row in table.rows:
        lines.append("| " + " | ".join(_cell_md(c, percent, decimals) for c in row) + " |")
    caption = _caption(table.caption, percent)
    if caption:
        lines.append("")
        lines.append(f"*{caption}*")
    return "\n".join(lines) + "\n"


def render_latex(table: Table, percent: bool = False, decimals: int | None = None) -> str:
    align = table.align or ("l" + "r" * (len(table.headers) - 1))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
    ]
    caption = _caption(table.caption, percent)
    if caption:
        lines.append(f"\\caption{{{_escape_tex(caption)}}}")
    if table.label:
        lines.append(f"\\label{{{table.label}}}")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_escape_tex(h) for h in table.headers) + " \\\\")
    lines.append("\\midrule")
    for row in table.rows:
        lines.append(" & ".join(_cell_tex(c, percent, decimals) for c in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"
