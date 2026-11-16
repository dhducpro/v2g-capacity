"""Generate the sensitivity table and figure from saved LP results."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "replication/sensitivity_checks"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
s = json.loads((HERE / "sensitivity_summary.json").read_text())
rows = list(csv.DictReader((HERE / "timing_energy_frontiers.csv").open()))
last_positive_q = max(
    (q for r in s["factorial_results"] for q, n in enumerate(r["positive_saving_counts_q0_to_q6"]) if n),
    default=0,
)
level_count = last_positive_q + 1
lines = [
    r"\begin{tabular}{@{}ll" + "r" * (1 + level_count) + r"@{}}",
    r"\toprule",
    rf" & & & \multicolumn{{{level_count}}}{{c}}{{Fleets with positive savings}}\\",
    rf"\cmidrule(lr){{4-{3 + level_count}}}",
    "Timing & Battery increment & Feasible (of 24) & "
    + " & ".join(f"$q={q}$" for q in range(level_count)) + r"\\",
    r"\midrule",
]
target_labels = {
    "reported_as_battery_increment": r"$d_i$",
    "grid_consistent_increment": r"$\eta_i^c d_i$",
}
grouped_targets = {}
for r in s["factorial_results"]:
    timing = "Hourly outward" if r["timing"] == "hourly_outward" else "Exact events"
    ns = r["positive_saving_counts_q0_to_q6"]
    key = (timing, r["feasible_fleet_count"], tuple(ns[:level_count]))
    grouped_targets.setdefault(key, []).append(target_labels[r["energy_mapping"]])
for (timing, feasible, counts), targets in grouped_targets.items():
    lines.append(
        f"{timing} & {' or '.join(targets)} & {feasible} & "
        + " & ".join(str(n) for n in counts)
        + r"\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "timing_energy_table.tex").write_text("\n".join(lines) + "\n")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.7), constrained_layout=False)
colors = {"reported_as_battery_increment": "#216b86", "grid_consistent_increment": "#b66032"}
style = {"hourly_outward": "--", "exact_events": "-"}
for ax, fleet, title in zip(
    axes,
    ["W203", "W206"],
    ["(a) W203: late charging bottleneck", "(b) W206: shorter connection windows"],
):
    for timing in ("hourly_outward", "exact_events"):
        for emap in ("reported_as_battery_increment", "grid_consistent_increment"):
            r = sorted(
                [
                    r
                    for r in rows
                    if r["fleet"] == fleet and r["timing"] == timing and r["energy_mapping"] == emap
                ],
                key=lambda r: int(r["q"]),
            )
            assert len(r) == 7 and all(x["status"] == "optimal" for x in r)
            lab = (
                ("Hourly" if timing == "hourly_outward" else "Exact events")
                + ", "
                + (
                    "unadjusted target"
                    if emap == "reported_as_battery_increment"
                    else "grid-consistent target"
                )
            )
            ax.plot(
                [int(x["q"]) for x in r],
                [max(0, float(x["saving_vs_same_variant_reservation_kw"])) for x in r],
                color=colors[emap],
                ls=style[timing],
                marker="o" if timing == "hourly_outward" else "s",
                markersize=3.7,
                linewidth=1.6,
                label=lab,
            )
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel("Protected withdrawals q")
    ax.set_ylabel("Capacity saving (kW)")
    ax.set_xticks(range(7))
    ax.set_xlim(-0.1, 6.1)
    ax.set_ylim(bottom=-0.05)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=2, frameon=False)
fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.27, wspace=0.26)
fig.savefig(OUT / "capacity_timing_sensitivity.pdf", bbox_inches="tight")
fig.savefig(OUT / "capacity_timing_sensitivity.png", dpi=200, bbox_inches="tight")
plt.close(fig)
(OUT / "timing_output_audit.json").write_text(
    json.dumps(
        {
            "source_rows": len(rows),
            "factorial_variants": len(s["factorial_results"]),
            "figure_fleets": ["W203", "W206"],
            "figure_values_per_fleet": 28,
            "zero_display_rule": "Negative numerical savings of magnitude below1e-6 clipped to zero for plotting.",
            "reoptimization": "Each saving uses reservation from the same timing/energy variant.",
        },
        indent=2,
    )
    + "\n"
)
print("Built timing/energy table and W203/W206 sensitivity figure.")
