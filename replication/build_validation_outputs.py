"""Build validation tables and one figure from completed, immutable results.

No solver is called. This script reads only saved comparison records, checks
their internal counts, and writes publication outputs plus a numerical audit.
Run from any directory: python replication/build_validation_outputs.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_presentation_figures import BLUE, GRAY, INK, ORANGE, TEAL, WIDTH, panel, tidy
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "replication" / "independent_validation"
RESULTS = STUDY / "results"
OUT = ROOT / "outputs"
TOL = 1e-6
COHORTS = ("bridge", "temporal", "mechanism")
LABELS = {"bridge": "Development", "temporal": "Temporal", "mechanism": "Synthetic"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def integer(value):
    return None if value in (None, "", "None") else int(value)


def group_summary(cases, frontier, cohort, n):
    group = [r for r in cases if r["cohort"] == cohort and int(r["N"]) == n]
    if not group:
        raise AssertionError((cohort, n, "Missing declared comparison group"))
    names = {r["case"] for r in group}
    observations = [r for r in frontier if r["cohort"] == cohort and r["case"] in names]
    qzero = [integer(r["q_zero"]) for r in group if integer(r["q_zero"]) is not None]
    matches = sum(
        integer(r["q_zero"]) is not None and integer(r["q_certificate"]) == integer(r["q_zero"])
        for r in group
    )
    sharper = sum(
        integer(r["q_certificate"]) is not None
        and integer(r["q_occupancy"]) is not None
        and integer(r["q_certificate"]) < integer(r["q_occupancy"])
        for r in group
    )
    return {
        "cohort": cohort,
        "N": n,
        "contexts": len(group),
        "q_records": len(observations),
        "positive_q0": sum(
            int(r["q"]) == 0 and r["status"] == "optimal" and float(r["saving_kw"]) > TOL
            for r in observations
        ),
        "positive_q1": sum(
            int(r["q"]) == 1 and r["status"] == "optimal" and float(r["saving_kw"]) > TOL
            for r in observations
        ),
        "cut_actual_threshold_matches": int(matches),
        "strict_occupancy_improvement": int(sharper),
        "q_zero_min": min(qzero) if qzero else None,
        "q_zero_max": max(qzero) if qzero else None,
        "unavailable_thresholds": len(group) - len(qzero),
        "case_names": sorted(names),
    }


def comparison_table(groups):
    show_matches = any(g["cut_actual_threshold_matches"] != g["contexts"] for g in groups)
    numeric_columns = 7 if show_matches else 6
    match_heading = r" & \shortstack{Threshold\\matches}" if show_matches else ""
    lines = [
        r"\begin{tabular}{@{}l" + "r" * numeric_columns + r"@{}}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Positive savings}" + " &" * (numeric_columns - 4) + r" \\",
        r"\cmidrule(lr){4-5}",
        r"Cohort & $N$ & Cases & $q=0$ & $q=1$ & \shortstack{Below occupancy\\threshold} & $q_{\rm zero}$ range" + match_heading + r"\\",
        r"\midrule",
    ]
    previous = None
    for row in groups:
        if previous is not None and previous != row["cohort"]:
            lines.append(r"\addlinespace[2pt]")
        low, high = row["q_zero_min"], row["q_zero_max"]
        interval = "--" if low is None else str(low) if low == high else f"{low}--{high}"
        label = LABELS[row["cohort"]] if previous != row["cohort"] else ""
        matches = f" & {row['cut_actual_threshold_matches']}" if show_matches else ""
        lines.append(
            f"{label} & {row['N']} & {row['contexts']} & "
            f"{row['positive_q0']} & {row['positive_q1']} & "
            f"{row['strict_occupancy_improvement']} & {interval}{matches}" + r"\\"
        )
        previous = row["cohort"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "validation_comparison_table.tex").write_text("\n".join(lines) + "\n")


def contract_table(rows):
    selected = sorted((r for r in rows if int(r["q"]) == 1), key=lambda r: float(r["H_kwh"]))
    recourse_values = {f"{float(r['restricted_recourse_M_kw']):.3f}" for r in selected}
    common_capacity = len(recourse_values) == 1
    columns = "rr" if common_capacity else "rrrr"
    headings = [r"Headroom $H$ (kWh)"]
    if not common_capacity:
        headings.extend([r"\shortstack{Fixed charging\\(kW)}", r"\shortstack{Cancel own recharge\\(kW)}"])
    headings.append("Extra capacity (kW)")
    lines = [r"\begin{tabular}{@{}" + columns + r"@{}}", r"\toprule", " & ".join(headings) + r"\\", r"\midrule"]
    audited = []
    headrooms_by_result = {}
    for row in selected:
        h, fixed, adapted, cost = (
            float(row[k])
            for k in ("H_kwh", "fixed_M_kw", "restricted_recourse_M_kw", "fixed_contract_cost_kw")
        )
        if abs(fixed - adapted - cost) > 1e-9 or cost < -TOL:
            raise AssertionError("Contract cost differs from its saved component capacities")
        capacities = () if common_capacity else (f"{fixed:.3f}", f"{adapted:.3f}")
        result = (*capacities, f"{max(0.0, cost):.3f}")
        headrooms_by_result.setdefault(result, []).append(f"{h:.2f}")
        audited.append(
            {
                "H_kwh": h,
                "q": 1,
                "fixed_M_kw": fixed,
                "restricted_recourse_M_kw": adapted,
                "contract_cost_kw": cost,
            }
        )
    for result, headrooms in headrooms_by_result.items():
        lines.append(" & ".join([", ".join(headrooms), *result]) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if common_capacity:
        capacity = next(iter(recourse_values))
        lines.append(r"\par\smallskip{\small Allowing the withdrawing donor to cancel its own recharge requires "
                     + f"{capacity}~kW at every headroom tested." + "}")
    (OUT / "contract_cost_table.tex").write_text("\n".join(lines) + "\n")
    return audited


def ratio_records(frontier):
    records = {cohort: [] for cohort in COHORTS}
    for row in frontier:
        if row["status"] != "optimal" or float(row["saving_kw"]) <= TOL:
            continue
        if row["bound_kw"] in ("", "None"):
            continue
        saving, bound = float(row["saving_kw"]), float(row["bound_kw"])
        ratio = bound / saving
        if not np.isfinite(ratio) or ratio < 1.0 - 1e-8:
            raise AssertionError((row["case"], row["q"], "Invalid positive bound ratio", ratio))
        key = f"{row['cohort']}|{row['case']}|{int(row['q'])}"
        u = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / (2**64 - 1)
        records[row["cohort"]].append(
            {
                "case": row["case"],
                "q": int(row["q"]),
                "N": int(row["N"]),
                "saving_kw": saving,
                "bound_kw": bound,
                "ratio": ratio,
                "display_ratio": max(1.0, ratio),
                "jitter": 0.44 * (u - 0.5),
            }
        )
    for values in records.values():
        values.sort(key=lambda r: (r["case"], r["q"]))
    return records


def make_figure(cases, ratios):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.25))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.24, wspace=0.38)
    threshold_records, centers = [], []
    offset = 0
    for n in (6, 12, 24):
        group = sorted(
            (r for r in cases if r["cohort"] == "temporal" and int(r["N"]) == n),
            key=lambda r: r["case"],
        )
        if len(group) != 12:
            raise AssertionError((n, "Expected twelve temporal contexts"))
        xs = np.arange(len(group), dtype=float) + offset
        for x, row in zip(xs, group):
            cut, actual, occupancy = (
                integer(row[k]) for k in ("q_certificate", "q_zero", "q_occupancy")
            )
            if cut is None or occupancy is None or cut != actual:
                raise AssertionError(
                    (row["case"], "Temporal cut/LP equality no longer holds; revise figure labels")
                )
            axes[0].plot([x, x], [cut / n, occupancy / n], color="#B6BEC3", linewidth=0.9, zorder=1)
            axes[0].scatter(
                [x],
                [occupancy / n],
                marker="s",
                s=20,
                facecolors="white",
                edgecolors=ORANGE,
                linewidths=0.9,
                zorder=3,
            )
            axes[0].scatter([x], [cut / n], marker="o", s=12, color=BLUE, zorder=4)
            threshold_records.append(
                {
                    "case": row["case"],
                    "N": n,
                    "x": float(x),
                    "q_cut": cut,
                    "q_zero": actual,
                    "q_occupancy": occupancy,
                    "cut_fraction": cut / n,
                    "occupancy_fraction": occupancy / n,
                }
            )
        centers.append(float(xs.mean()))
        offset += len(group) + 3
    axes[0].set(
        xlim=(-1, offset - 3),
        ylim=(-0.035, 1.035),
        ylabel="Withdrawal fraction",
        xticks=centers,
        xticklabels=[r"$N=6$", r"$N=12$", r"$N=24$"],
        yticks=[0, 0.25, 0.5, 0.75, 1],
    )
    axes[0].legend(
        handles=[
            Line2D(
                [], [], marker="o", color=BLUE, linestyle="none", markersize=4, label="Cut = LP"
            ),
            Line2D(
                [],
                [],
                marker="s",
                markerfacecolor="white",
                markeredgecolor=ORANGE,
                linestyle="none",
                markersize=4.5,
                label="Occupancy",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        frameon=False,
        fontsize=8,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    panel(axes[0], "(a)", "Temporal thresholds")
    tidy(axes[0])

    ratio_summary = {}
    for position, (cohort, color) in enumerate(zip(COHORTS, (GRAY, BLUE, TEAL))):
        rows = ratios[cohort]
        values = np.array([r["display_ratio"] for r in rows])
        if not len(rows) or values.max() > 500:
            raise AssertionError((cohort, "Figure range must include every positive-bound ratio"))
        xs = np.array([position + r["jitter"] for r in rows])
        axes[1].scatter(
            xs, values, s=11, color=color, alpha=0.58, edgecolors="none", zorder=3, clip_on=False
        )
        median = float(np.median(values))
        axes[1].plot(
            [position - 0.20, position + 0.20],
            [median, median],
            color=INK,
            linewidth=1.8,
            zorder=5,
            solid_capstyle="butt",
            clip_on=False,
        )
        raw = np.array([r["ratio"] for r in rows])
        ratio_summary[cohort] = {
            "count": len(rows),
            "min": float(raw.min()),
            "median": float(np.median(raw)),
            "max": float(raw.max()),
            "plotted_records": rows,
        }
    bridge_values = ratios["bridge"]
    maximum = max(bridge_values, key=lambda r: r["ratio"])
    axes[1].annotate(
        f"{maximum['ratio']:.1f}",
        xy=(maximum["jitter"], maximum["display_ratio"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7.5,
        color=INK,
        ha="left",
        va="bottom",
    )
    axes[1].set_yscale("log")
    axes[1].set(
        xlim=(-0.5, 2.5),
        ylim=(0.8, 500),
        xticks=[0, 1, 2],
        xticklabels=[f"{LABELS[c]}\n({len(ratios[c])} values)" for c in COHORTS],
        ylabel="Bound / positive saving",
    )
    axes[1].yaxis.set_major_locator(FixedLocator([1, 10, 100, 500]))
    axes[1].yaxis.set_major_formatter(ScalarFormatter())
    axes[1].yaxis.set_minor_locator(NullLocator())
    axes[1].legend(
        handles=[Line2D([], [], color=INK, linewidth=1.8, label="Median")],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    panel(axes[1], "(b)", "Positive bound tightness")
    tidy(axes[1])
    fig.savefig(
        OUT / "presentation_validation.pdf",
        metadata={
            "Title": "Critical-cut validation and positive-bound tightness",
            "Subject": "Saved temporal thresholds and descriptive repeated fleet-q ratios; no statistical intervals",
            "Creator": "build_validation_outputs.py",
        },
    )
    fig.savefig(OUT / "presentation_validation.png", dpi=240)
    plt.close(fig)
    return {
        "temporal_threshold_records": threshold_records,
        "positive_bound_ratios": ratio_summary,
        "ratio_display_rule": "Ratios below1 only within1e-8 floating tolerance are drawn at1; every positive-saving observation is plotted without upper clipping",
        "jitter_rule": "SHA256(cohort|case|q), first64bits mapped to[-0.22,0.22]; deterministic and independent of ratio",
        "statistics": "Descriptive medians only; repeated q levels share fleets, and synthetic variants share base configurations; no confidence intervals",
    }


def main():
    paths = [
        RESULTS / "frontiers.csv",
        RESULTS / "case_summary.csv",
        RESULTS / "contract_cost.csv",
        RESULTS / "validation_summary.json",
        RESULTS / "contract_cost_audit.json",
        STUDY / "validation_protocol.json",
    ]
    case_paths = sorted((RESULTS / "cases").glob("*.json"))
    if len(case_paths) != 156:
        raise AssertionError("Expected all156 saved per-context records")
    paths.extend(case_paths)
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    frontier, cases, contract = (read_csv(p) for p in paths[:3])
    saved = json.loads(paths[3].read_text())
    if len(frontier) != 2148 or len(cases) != 156:
        raise AssertionError("Saved study size differs from the completed frozen design")
    keys = {(r["cohort"], r["case"], int(r["q"])) for r in frontier}
    if len(keys) != len(frontier):
        raise AssertionError("Duplicate fleet-q observations")
    for case in cases:
        expected = {(case["cohort"], case["case"], q) for q in range(int(case["N"]) + 1)}
        if not expected <= keys:
            raise AssertionError((case["case"], "Missing protected levels"))
    threshold_checks = []
    for case in cases:
        record = json.loads((RESULTS / "cases" / (case["case"] + ".json")).read_text())
        cuts = record["cut_extraction"]["cuts"]
        strict = min((int(cut["q_certificate"]) for cut in cuts), default=None)
        numerical = integer(case["q_certificate"])
        if numerical != record["summary"]["q_certificate"]:
            raise AssertionError((case["case"], "CSV/JSON numerical threshold mismatch"))
        threshold_checks.append(
            {
                "case": case["case"],
                "cohort": case["cohort"],
                "strict_donor_count_threshold": strict,
                "numerical_corrected_bound_threshold": numerical,
                "matches": strict is not None and strict == numerical,
            }
        )
    threshold_matches = sum(r["matches"] for r in threshold_checks)
    if threshold_matches != 156:
        raise AssertionError(
            "Strict donor-count and numerical corrected-bound thresholds differ; revise their common display"
        )
    groups = [
        group_summary(cases, frontier, c, n)
        for c, n in [
            ("bridge", 6),
            ("temporal", 6),
            ("temporal", 12),
            ("temporal", 24),
            ("mechanism", 6),
            ("mechanism", 12),
            ("mechanism", 24),
        ]
    ]
    ratios = ratio_records(frontier)
    for cohort in COHORTS:
        selected = [g for g in groups if g["cohort"] == cohort]
        check = saved["cohorts"][cohort]
        checks = {
            "contexts": "contexts",
            "q_records": "frontier_calls",
            "positive_q0": "positive_at_q0",
            "positive_q1": "positive_at_q1",
            "cut_actual_threshold_matches": "exact_collapse_matches",
            "strict_occupancy_improvement": "strictly_improves_occupancy",
        }
        for local, source in checks.items():
            if sum(g[local] for g in selected) != check[source]:
                raise AssertionError((cohort, local, "Disagreement with saved summary"))
        values = np.array([r["ratio"] for r in ratios[cohort]])
        if len(values) != check["positive_bound_ratio_count"] or not np.allclose(
            [values.min(), np.median(values), values.max()],
            check["positive_bound_ratio_min_median_max"],
            rtol=1e-10,
            atol=1e-8,
        ):
            raise AssertionError((cohort, "Ratio summary mismatch"))
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.bbox": None,
        }
    )
    comparison_table(groups)
    contract_rows = contract_table(contract)
    figure = make_figure(cases, ratios)
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    if before != after:
        raise AssertionError("A saved source changed during output generation")
    audit = {
        "scope": "Saved-results presentation rebuild only; no optimizer calls or input changes",
        "optimizer_calls": 0,
        "source_sha256": before,
        "frozen_sources_unchanged": True,
        "frontier_q_records": len(frontier),
        "contexts": len(cases),
        "new_temporal_contexts": 36,
        "new_synthetic_contexts": 96,
        "retrospective_bridge_contexts": 24,
        "comparison_groups": groups,
        "contract_q1_rows": contract_rows,
        "contract_all_q_records": len(contract),
        "contract_fixed_LPs": 25,
        "contract_restricted_LPs": 25,
        "canvas_width_mm": 178,
        "axis_font_pt": 9,
        "ticks_font_pt": 8.5,
        "saving_tolerance_kw": TOL,
        "figure": figure,
        "count_numeric_threshold_matches": threshold_matches,
        "count_numeric_threshold_definitions": "Strict: minimum stored cut donor-count threshold using e>0/Pd>0. Numerical: first q whose corrected upper bound is<=1e-6kW. These are distinct definitions, checked to coincide on every saved context.",
        "count_numeric_threshold_checks": threshold_checks,
        "output_files": [
            "outputs/validation_comparison_table.tex",
            "outputs/contract_cost_table.tex",
            "outputs/presentation_validation.pdf",
            "outputs/presentation_validation.png",
        ],
    }
    (OUT / "validation_output_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        "Built two validation tables and one178mm figure from156contexts/2148saved frontier rows; sources unchanged."
    )


if __name__ == "__main__":
    main()
