"""Build only the figures/tables retained in the focused capacity manuscript.

No optimizer is called. Fleet plots use retained independent LP results.
The restricted-class plot is an analytical illustration, explicitly labeled
as such; it is not an additional data experiment.
"""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "replication/capacity_checks"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

rows = list(csv.DictReader((CHECKS / "full_frontier_independent_lp.csv").open()))
fleets = sorted({r["fleet"] for r in rows})
gap = np.zeros((len(fleets), 7))
for r in rows:
    gap[fleets.index(r["fleet"]), int(r["q"])] = max(0, float(r["saving_vs_reservation_kw"]))
gap[gap < 1e-6] = 0
assert len(rows) == 168 and len(fleets) == 24
counts = [(gap[:, q] > 1e-6).sum() for q in range(7)]
assert counts == [13, 3, 2, 2, 0, 0, 0]
lines = [
    r"\begin{tabular}{rrrrrrr}",
    r"\toprule",
    r"$q$ & Positive & Zero & Certified zero & Mean $\Delta M$ & Median $\Delta M$ & Max. $\Delta M$\\",
    r"\midrule",
]
for q in range(7):
    rs = [r for r in rows if int(r["q"]) == q]
    certified = sum(q >= int(r["cut_collapse_q"]) for r in rs)
    vals = gap[:, q]
    lines.append(
        f"{q} & {counts[q]} & {24 - counts[q]} & {certified} & {vals.mean():.4f} & {np.median(vals):.4f} & {vals.max():.4f}"
        + r"\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "capacity_frontier_table.tex").write_text("\n".join(lines) + "\n")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)
fig, ax = plt.subplots(
    1, 2, figsize=(7.25, 4.25), gridspec_kw={"width_ratios": [1.1, 1]}, layout="constrained"
)
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("#eeeeee")
im = ax[0].imshow(
    np.ma.masked_equal(gap, 0), aspect="auto", interpolation="none", cmap=cmap, norm=Normalize(0, 6)
)
ax[0].set(
    xticks=range(7),
    yticks=range(24),
    yticklabels=fleets,
    xlabel="Protected withdrawals q",
    title="(a) All 24 fleet contexts",
)
ax[0].tick_params(axis="y", labelsize=7.5)
cb = fig.colorbar(im, ax=ax[0], fraction=0.06, pad=0.025)
cb.set_label("Capacity saving (kW)")
styles = [("W104", "#2675b8", "o"), ("W203", "#dd7b22", "s"), ("W206", "#168a79", "^")]
for f, c, m in styles:
    ax[1].plot(range(7), gap[fleets.index(f)], color=c, marker=m, ms=4, lw=1.6, label=f)
ax[1].set(
    xlabel="Protected withdrawals q",
    ylabel="Capacity saving (kW)",
    xticks=range(7),
    ylim=(-0.1, 6.1),
    title="(b) Fleets surviving q = 1",
)
ax[1].grid(alpha=0.2)
ax[1].legend(frameon=False)
fig.savefig(OUT / "capacity_frontier.pdf")
fig.savefig(OUT / "capacity_frontier.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.25, 2.65), layout="constrained")
k, D, n, q = 0.81, 4.0, 4, 1
gamma = 1 - q / n
B = np.linspace(0, 5, 251)
power = gamma * B
recharge = k * gamma * D / (1 + k * gamma)
ax[0].plot(B, power, color="#9c6fb6", ls="--", lw=1.3, label="Surviving power limit")
ax[0].axhline(recharge, color="#777777", ls=":", lw=1.5, label="Replenishment limit")
ax[0].plot(B, np.minimum(power, recharge), color="#168a79", lw=2, label="Exact capacity saving")
ax[0].set(
    xlabel="Total donor discharge rating B (kW)",
    ylabel="Capacity saving (kW)",
    title="(a) Power and replenishment",
    xlim=(0, 5),
    ylim=(0, 3.9),
)
ax[0].legend(frameon=False, loc="upper left", fontsize=7)
ns = np.arange(1, 9)
for q, c, m in [(0, "#168a79", "^"), (1, "#dd7b22", "s"), (2, "#2675b8", "o")]:
    g = np.maximum(0, 1 - q / ns)
    values = np.minimum(g * 3.0, k * g * D / (1 + k * g))
    ax[1].plot(ns, values, color=c, marker=m, ms=3.5, lw=1.5, label=f"q = {q}")
ax[1].set(
    xlabel="Number of donors n",
    ylabel="Capacity saving (kW)",
    title="(b) Fixed aggregate resources",
    xticks=ns,
    ylim=(0, 2.05),
)
ax[1].legend(frameon=False, loc="lower right", fontsize=8)
for a in ax:
    a.grid(alpha=0.2)
fig.savefig(OUT / "capacity_sharp_class.pdf")
fig.savefig(OUT / "capacity_sharp_class.png", dpi=180)
plt.close(fig)

summary = {
    "fleet_count": len(fleets),
    "positive_saving_counts": list(map(int, counts)),
    "plot_basis": "Stored independent all-q LP results; analytical second figure is not a new experiment",
    "numerical_tolerance_kw": 1e-6,
}
(OUT / "capacity_output_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
