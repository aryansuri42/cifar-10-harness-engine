"""Article figures from the [REPORT] lines in logs/run_*.log.

    python plot_progress.py   -> figures/outcomes.png, figures/versions.png, figures/timeline.png
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

TARGET = 0.95  # keep in sync with harness.py (importing it would start a new run log)
ROOT = Path(__file__).parent
OUT = ROOT / "figures"
# log -> harness version from the README results table; unknown logs fall back to their date
VERSIONS = {"run_20260922_135406": "Harness v3", "run_20260924_072443": "Harness v5"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
GRADED, REJECTED, CRASHED = "#2a78d6", "#fab219", "#d03b3b"
ROW = re.compile(r"\[REPORT\] attempt (\d+) \| (.*?)\s*\| verified_acc=\s*([\d.]+|-)")
TOTAL = re.compile(r"\[REPORT\] model=.*total time ([\d.]+) min")

plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 14,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 14})


def outcome(text):
    return "rejected" if text.startswith("static check") else "crashed" if text.startswith("crashed") else "graded"


def parse(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = [(int(n), outcome(t), None if a == "-" else float(a)) for n, t, a in ROW.findall(text)]
    total = TOTAL.search(text)
    return {"name": VERSIONS.get(path.stem, path.stem[4:12]), "rows": rows,
            "minutes": float(total.group(1)) if total else None}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    print("saved", (OUT / name).relative_to(ROOT).as_posix())


def outcomes(runs):
    """Where every LLM generation ended up: rejected before training, crashed, or trained and graded."""
    fig, ax = plt.subplots(figsize=(10, 1.4 + 1.1 * len(runs)))
    for i, r in enumerate(runs):
        left = 0
        for kind, color in (("graded", GRADED), ("rejected", REJECTED), ("crashed", CRASHED)):
            n = sum(o == kind for _, o, _ in r["rows"])
            if n:
                ax.barh(i, n, left=left, color=color, height=0.55, edgecolor="white", lw=2,
                        label=kind if i == 0 or kind not in ax.get_legend_handles_labels()[1] else None)
                ax.text(left + n / 2, i, str(n), ha="center", va="center", fontweight="bold",
                        color="white" if kind != "rejected" else INK)
                left += n
        best = max((a for _, _, a in r["rows"] if a is not None), default=0)
        mins = f" in {r['minutes']:.0f} min" if r["minutes"] else ""
        ax.text(left + 0.4, i, f"{len(r['rows'])} generations{mins}\nbest {best:.4f}", va="center", color=MUTED)
    ax.set_yticks(range(len(runs)), [r["name"] for r in runs], color=INK, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(len(r["rows"]) for r in runs) * 1.3)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.legend(["trained & graded", "rejected by static checks", "crashed"], frameon=False, ncol=3,
              loc="upper left", bbox_to_anchor=(0, -0.02))
    ax.set_title("Same 7B model, better harness: fewer wasted attempts")
    save(fig, "outcomes.png")


def versions():
    """Best verified accuracy per harness version (README results table; v4 was never completed)."""
    data = [("v1", "basic loop", 0.10), ("v2", "+ scaling check\n+ overfit gap", 0.7731),
            ("v3", "+ duplicate detection\n+ best-so-far anchor", 0.8759),
            ("v5", "+ reference script\n+ synced grader\n+ harness picks experiment", 0.9758)]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = range(len(data))
    ax.bar(x, [a for *_, a in data], color=GRADED, width=0.55)
    for i, (_, _, a) in enumerate(data):
        ax.text(i, a + 0.015, f"{a:.4f}" if a > 0.2 else "0.10", ha="center", color=INK, fontweight="bold")
    ax.axhline(TARGET, color=MUTED, ls="--", lw=1)
    ax.text(-0.4, TARGET + 0.012, f"target {TARGET:.2f}", color=MUTED)
    ax.set_xticks(x, [f"{v}\n{d}" for v, d, _ in data], color=INK, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("best verified test accuracy")
    ax.set_title("Qwen2.5-Coder-7B on CIFAR-10: only the harness changed")
    style(ax)
    save(fig, "versions.png")


def timeline(runs):
    """Every generation in order: graded ones at their accuracy, failures on a strip at the bottom."""
    n = max(len(r["rows"]) for r in runs)
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.6 + 2.4 * len(runs)), sharex=True, squeeze=False)
    floor = 0.6
    for ax, r in zip(axes[:, 0], runs):
        best, xs, ys = 0, [], []
        for num, kind, acc in r["rows"]:
            if kind == "graded":
                ax.plot(num, acc, "o", color=GRADED, ms=8, mec="white", mew=1.5, zorder=3)
                if acc > best:
                    ax.annotate(f"{acc:.4f}", (num, acc), xytext=(0, 9), textcoords="offset points",
                                ha="center", color=INK, fontsize=10)
                best = max(best, acc)
            else:
                ax.plot(num, floor, "X" if kind == "crashed" else "v", ms=9, zorder=3,
                        color=CRASHED if kind == "crashed" else REJECTED, mec="white", mew=1)
            if best:
                xs.append(num), ys.append(best)
        ax.step(xs + [xs[-1] + 0.5], ys + [ys[-1]], where="post", color=GRADED, lw=2, alpha=0.5)
        ax.axhline(TARGET, color=MUTED, ls="--", lw=1)
        ax.text(n + 0.8, TARGET + 0.008, f"target {TARGET:.2f}", ha="right", va="bottom", color=MUTED, fontsize=10)
        ax.axhspan(floor - 0.035, floor + 0.035, color="#f0efec", zorder=0)
        ax.set_ylim(floor - 0.05, 1.02)
        ax.set_yticks([floor, 0.7, 0.8, 0.9, 1.0], ["failed", "0.70", "0.80", "0.90", "1.00"])
        ax.set_ylabel("test accuracy")
        ax.set_title(r["name"], fontsize=12)
        style(ax)
    ax.set_xlim(0.3, n + 0.9)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("LLM generation")
    handles = [plt.Line2D([], [], marker=m, ls="", color=c, ms=9) for m, c in
               (("o", GRADED), ("v", REJECTED), ("X", CRASHED))]
    handles.append(plt.Line2D([], [], color=GRADED, lw=2, alpha=0.5))
    fig.legend(handles, ["graded", "rejected by static checks", "crashed", "best so far"], frameon=False,
               ncol=4, loc="lower left", bbox_to_anchor=(0.06, -0.04))
    fig.suptitle("Every attempt, in order", x=0.06, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "timeline.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    runs = [r for p in sorted(ROOT.glob("logs/run_*.log")) if (r := parse(p))["rows"]]
    outcomes(runs)
    versions()
    timeline(runs)
