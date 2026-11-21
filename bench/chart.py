"""Ladder chart from a bench/results JSON: one bar per measured rung (studies per second, log scale), stdlib SVG.

    uv run python -I bench/chart.py bench/results/ladder-<hw>.json   # writes the .svg beside it
"""
from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path


def svg(res: dict) -> str:
    measured = {k: v for k, v in res["rungs"].items() if isinstance(v, dict)}
    pending = [k for k, v in res["rungs"].items() if isinstance(v, str)]
    if not measured:
        raise ValueError("no measured rung: nothing to chart")
    W, row, left, top = 720, 44, 170, 56
    H = top + row * (len(measured) + len(pending)) + 48
    vals = {k: v["studies_per_s"] for k, v in measured.items()}
    lo = math.floor(math.log10(min(vals.values()))) - 1
    hi = math.ceil(math.log10(max(vals.values()))) + 1
    x = lambda v: left + (W - left - 90) * (math.log10(v) - lo) / (hi - lo)
    base = min(vals.values())
    hw = res["hardware"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="13">',
           '<text x="12" y="24" font-size="16" font-weight="bold">Speedup ladder: studies per second (log scale)</text>',
           (f'<text x="12" y="44" fill="#555">CPU {escape(hw["cpu"])}, {hw["logical_cpus"]} logical; '
           f'GPU {escape(hw["gpu"] or "none")}; data {escape(res["data"])}</text>')]
    y = top
    for k, v in vals.items():
        out += [f'<text x="12" y="{y + 22}">{escape(k)}</text>',
                f'<rect x="{left}" y="{y + 8}" width="{x(v) - left:.1f}" height="22" fill="#2f7d4f"/>',
                f'<text x="{x(v) + 6:.1f}" y="{y + 24}">{v:.3g}/s ({v / base:.3g}x)</text>']
        y += row
    for k in pending:
        out += [f'<text x="12" y="{y + 22}" fill="#888">{escape(k)}</text>',
                f'<text x="{left}" y="{y + 22}" fill="#888">pending, not measured</text>']
        y += row
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = src.with_suffix(".svg")
    dst.write_text(svg(json.loads(src.read_text())))
    print(dst)
