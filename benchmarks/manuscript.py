"""Layout of the published benchmark figures, which the reproduced figures reuse.

* ``geometry(key)``  size of the published figure, its axes rectangles and their y range, font
                     sizes.  Read from figure_geometry.json, which was extracted from the
                     matplotlib .pgf/.pdf files included by the paper (figure ``key`` = its number
                     in the paper, ``label`` = its \\label, ``source`` = the file it was read from).
                     The reproduction plots reuse it so that both versions have the same size,
                     fonts and inches per decade on the y axis.
* ``FONT_PREAMBLE``  the paper's fonts for the pgf backend.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEOMETRY = HERE / "figure_geometry.json"

# Fonts of neurips_2026.sty (Times text, Helvetica sans); the published .pgf files use \sffamily.
FONT_PREAMBLE = (r"\usepackage[T1]{fontenc}\renewcommand{\rmdefault}{ptm}\renewcommand{\sfdefault}{phv}"
                 r"\usepackage{amsmath}\def\mathdefault#1{#1}")


@dataclass
class Axis:
    rect: tuple            # x0, y0, w, h in inches
    yscale: str = "linear"
    ylim: tuple | None = None
    yticks: list = field(default_factory=list)   # (value, y in inches)

    @property
    def decades(self):
        return math.log10(self.ylim[1] / self.ylim[0]) if self.yscale == "log" and self.ylim else None


@dataclass
class Geometry:
    figsize: tuple
    axes: list
    tick_size: float
    label_size: float


def _from_dict(d) -> Geometry:
    return Geometry(figsize=tuple(d["figsize"]), tick_size=d["tick_size"], label_size=d["label_size"],
                    axes=[Axis(rect=tuple(a["rect"]), yscale=a["yscale"], ylim=tuple(a["ylim"]) if a["ylim"] else None,
                               yticks=[tuple(t) for t in a["yticks"]]) for a in d["axes"]])


@lru_cache(None)
def geometry(key: str) -> Geometry | None:
    try:
        d = json.loads(GEOMETRY.read_text())[key]
    except (OSError, KeyError, ValueError):
        return None
    return _from_dict(d)
