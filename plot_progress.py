"""Plot verified test accuracy per graded attempt, one line per run, from the [METRIC] lines in logs/run_*.log.

    python plot_progress.py          -> logs/progress.png
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

TARGET = 0.95  # keep in sync with harness.py (importing it would start a new run log)

ROOT = Path(__file__).parent
METRIC = re.compile(r"\[METRIC\] test_acc=([\d.]+)")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def parse(path):
    return [float(m.group(1)) for m in METRIC.finditer(path.read_text(encoding="utf-8", errors="replace"))]


runs = {p.stem.removeprefix("run_"): accs for p in sorted(ROOT.glob("logs/run_*.log")) if (accs := parse(p))}

fig, ax = plt.subplots(figsize=(8, 4.5))
for (name, accs), color in zip(runs.items(), COLORS):  # ponytail: 8 runs max, fold older ones if you keep more
    x = range(1, len(accs) + 1)
    best = [max(accs[:i]) for i in x]
    ax.plot(x, accs, "o", color=color, alpha=0.45, markersize=6)
    ax.step(x, best, where="post", color=color, lw=2, label=f"{name}  (best {best[-1]:.4f})")
    ax.annotate(f"{best[-1]:.4f}", (len(accs), best[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color="#0b0b0b")

ax.axhline(TARGET, color="#52514e", ls="--", lw=1)
ax.text(1, TARGET, f" target {TARGET:.2f}", va="bottom", color="#52514e")
ax.set(xlabel="graded attempt", ylabel="verified test accuracy",
       title="Verified CIFAR-10 test accuracy per attempt (dots = attempt, line = best so far)")
ax.xaxis.get_major_locator().set_params(integer=True)
ax.grid(axis="y", color="#e5e4e0", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
out = ROOT / "logs" / "progress.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
