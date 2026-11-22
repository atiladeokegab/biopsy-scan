"""Plot accepted points from one or more receipts for the same archived H100 design."""
import argparse
import json
from collections import defaultdict
from math import isclose, isfinite
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

GPU_COUNTS = (1, 2, 4, 8)


def verified_timing(point: dict, size: int, total: int, seen_tasks: set[str]) -> dict:
    workers = point["workers"]
    tasks = {w["gpu"]["modal_task_id"] for w in workers}
    if (size not in GPU_COUNTS or point["gpus"] != size or len(workers) != size
            or len(tasks) != size or not all(tasks) or not seen_tasks.isdisjoint(tasks)):
        raise ValueError(f"missing or duplicate worker task IDs for {size} GPUs")
    starts, ends = zip(*(w["compute_window"] for w in workers), strict=True)
    span, overlap = max(ends) - min(starts), min(ends) - max(starts)
    wall = point["end_to_end_wall_s"]
    if not all(isfinite(t) and t > 0 for t in (*starts, *ends, overlap, span, wall)) or span > wall:
        raise ValueError(f"invalid compute windows or wall time for {size} GPUs")
    throughput = total * 3600 / span
    expected = {"compute_span_s": span, "common_overlap_s": overlap,
                "wall_s_per_evaluated_study": wall / total, "studies_per_hour_compute_span": throughput}
    if any(not isclose(point[key], value, rel_tol=1e-9) for key, value in expected.items()):
        raise ValueError(f"recorded timing disagrees with worker windows for {size} GPUs")
    seen_tasks.update(tasks)
    return point | expected


def measured_points(paths: list[Path]) -> tuple[dict, dict]:
    receipts = [json.loads(p.read_text()) for p in paths]
    # The runner checkpoints each accepted point while other groups are still in flight.
    if any(r["status"] not in ("INCOMPLETE", "PARTIAL", "MEASURED") for r in receipts):
        raise ValueError("unrecognized benchmark receipt state")
    fields = ("reference", "reference_commit", "seed_root", "synthetic_outcome_plan", "hardware_request")
    design = {k: receipts[0][k] for k in fields}
    if any(any(r[k] != v for k, v in design.items()) for r in receipts):
        raise ValueError("receipts describe different designs or hardware requests")
    plan = design["synthetic_outcome_plan"]
    total = sum(plan.values())
    points = defaultdict(list)
    seen_tasks = set()
    for r in receipts:
        for repeat in r["repeats"]:
            for size, point in repeat.items():
                parity = point["parity"]
                if (point["status"] != "MEASURED" or point["evaluated_studies"] != total
                        or point["summary"]["studies"] != plan or not parity["x_bitwise"]
                        or parity["studies_compared"] != total or not 0 <= parity["max_abs_maxT_delta"] <= 1e-9
                        or not all(point[k] for k in ("all_overlapped", "seed_coverage_exact", "distinct_task_ids"))):
                    raise ValueError(f"invalid measured point for {size} GPUs")
                n = int(size)
                points[n].append(verified_timing(point, n, total, seen_tasks))
    if not points:
        raise ValueError("no measured points")
    return plan, points


def plot(paths: list[Path], out: Path, subtitle: str = "") -> None:
    plan, points = measured_points(paths)
    sizes = sorted(points)
    ticks = GPU_COUNTS
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    title = f"Fixed Charville design on Modal H100s · {sum(plan.values()):,} evaluations"
    if subtitle:
        title += f"\n{subtitle}"
    elif len(paths) > 1:
        title += "\nCross-launch observations"
    fig.suptitle(title, fontweight="bold")
    for ax, field, title, unit in (
        (axes[0], "studies_per_hour_compute_span", "Compute throughput", "Evaluations per hour"),
        (axes[1], "end_to_end_wall_s", "End-to-end time", "Wall seconds"),
    ):
        values = [[p[field] for p in points[n]] for n in sizes]
        mid = [median(v) for v in values]
        bounds = [[m - min(v) for m, v in zip(mid, values)], [max(v) - m for m, v in zip(mid, values)]]
        if 1 in points and ax is axes[0]:
            ax.plot(ticks, [mid[0] * n for n in ticks], "--", color="#606872", label="Ideal linear reference")
        ax.errorbar(sizes, mid, yerr=bounds, fmt="o", capsize=4, color="#007d69", label="Measured median and min–max")
        ax.set_xscale("log", base=2)
        ax.set(title=title, xlabel="H100 workers", ylabel=unit, ylim=(0, None))
        ax.set_xticks(ticks, [str(n) for n in ticks])
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", alpha=.22)
        ax.legend(loc="upper left" if ax is axes[0] else "upper right", fontsize=8)
    counts = " · ".join(f"{n} ({len(points[n])})" for n in sizes)
    fig.supxlabel(f"GPU count (verified runs)   {counts}   |   End-to-end includes setup, transfer and global threshold", fontsize=8)
    fig.savefig(out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subtitle", default="")
    args = parser.parse_args()
    plot(args.receipts, args.out, args.subtitle)
