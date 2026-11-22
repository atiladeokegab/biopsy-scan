"""Plot a completed Modal fleet receipt: python bench/plot_fleet_scaling.py RECEIPT.json."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def plot(receipt: Path) -> Path:
    data = json.loads(receipt.read_text())
    if data["status"] != "MEASURED" or data["pilot"] or len(data["repeats"]) < 2:
        raise ValueError("plot requires a complete measured fleet sweep with repeats")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    fig.suptitle("Charville re-imaging on Modal A100-40GB workers", fontweight="bold")
    for ax, mode, title in zip(axes, ("strong", "weak"), (
        f"Strong scaling · {data['strong_studies']} fixed studies",
        f"Weak scaling · {data['weak_studies_per_gpu']} studies per GPU",
    ), strict=True):
        rows = data["summary"][mode]
        gpus = sorted(map(int, rows))
        points = [rows[str(n)]["studies_per_hour"] for n in gpus]
        mid = [p["median"] for p in points]
        bounds = [[p["median"] - p["min"] for p in points], [p["max"] - p["median"] for p in points]]
        ax.plot(gpus, [mid[0] * n for n in gpus], "--", color="#606872", label="Ideal linear reference")
        ax.errorbar(gpus, mid, yerr=bounds, fmt="o-", capsize=4, color="#007d69", label="MEASURED · median / min–max")
        ax.set(title=title, xlabel="GPU workers", ylabel="Studies per hour", xticks=gpus, ylim=(0, None))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", alpha=.22)
        ax.legend(loc="upper left", fontsize=8)
    fig.supxlabel(f"{len(data['repeats'])} repeats · Compute span includes staggered shard starts · Setup and transfer reported separately",
                 fontsize=9)
    out = receipt.with_suffix(".svg")
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(plot(parser.parse_args().receipt))
