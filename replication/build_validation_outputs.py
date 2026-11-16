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
from build_presentation_figures import BLUE, GRAY, INK, ORANGE, WIDTH, tidy
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, PercentFormatter

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
    fig, (threshold_ax, ratio_ax) = plt.subplots(1, 2, figsize=(WIDTH, 3.02))
    fig.subplots_adjust(left=0.102, right=0.97, top=0.78, bottom=0.19, wspace=0.51)
    threshold_records, centers = [], []
    offset = 0
    for n in (6, 12, 24):
        group = [r for r in cases if r["cohort"] == "temporal" and int(r["N"]) == n]
        if len(group) != 12:
            raise AssertionError((n, "Expected twelve temporal contexts"))
        for row in group:
            cut, actual, occupancy = (integer(row[k]) for k in ("q_certificate", "q_zero", "q_occupancy"))
            if cut is None or occupancy is None or cut != actual:
                raise AssertionError((row["case"], "Temporal cut/LP equality no longer holds; revise figure labels"))
        group.sort(key=lambda r: (int(r["q_certificate"]), int(r["q_occupancy"]), r["case"]))
        xs = np.arange(len(group)) + offset
        cut = np.array([int(r["q_certificate"]) for r in group])
        occupancy = np.array([int(r["q_occupancy"]) for r in group])
        threshold_ax.vlines(xs, cut / n, occupancy / n, color="#ABB8BE", linewidth=0.85)
        threshold_ax.scatter(xs, occupancy / n, marker="s", s=17, facecolors="white",
                             edgecolors=ORANGE, linewidths=0.9, zorder=3)
        threshold_ax.scatter(xs, cut / n, s=10, color=BLUE, linewidths=0, zorder=4)
        for x, row, q_cut, q_occ in zip(xs, group, cut.tolist(), occupancy.tolist()):
            threshold_records.append({
                "case": row["case"], "N": n, "x": float(x), "q_cut": q_cut, "q_zero": q_cut,
                "q_occupancy": q_occ, "cut_fraction": q_cut / n, "occupancy_fraction": q_occ / n,
            })
        centers.append(float(xs.mean()))
        offset += len(group) + 3
    threshold_ax.set(xlim=(-1, offset - 3), ylim=(-0.045, 1.045), xticks=centers,
                     xticklabels=["6 EVs", "12 EVs", "24 EVs"], yticks=[0, 0.25, 0.5, 0.75, 1],
                     xlabel="12 fleets per group", ylabel=r"Withdrawal threshold $q/N$")
    threshold_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    threshold_ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, linestyle="none", markersize=4, label="Certified / observed"),
        Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=ORANGE,
               linestyle="none", markersize=4, label="Occupancy only"),
    ], loc="lower left", bbox_to_anchor=(-0.02, 1.005), borderaxespad=0, ncol=2,
       frameon=False, fontsize=8, columnspacing=0.6, handletextpad=0.2)
    threshold_ax.set_title("(a)  When savings disappear", loc="left", y=1.18, fontsize=9, fontweight="bold")
    tidy(threshold_ax)

    ratio_summary = {}
    for position, cohort in enumerate(COHORTS):
        rows = ratios[cohort]
        values = np.array([r["display_ratio"] for r in rows])
        raw = np.array([r["ratio"] for r in rows])
        offsets = [position + r["jitter"] * 0.55 for r in rows]
        ratio_ax.scatter(values, offsets, s=12, color=BLUE, alpha=0.52, linewidths=0, zorder=3)
        median = float(np.median(raw))
        ratio_ax.vlines(max(1, median), position - 0.15, position + 0.15,
                        color=INK, linewidth=1.8, zorder=4)
        ratio_ax.annotate(f"Median {median:.2f}×", (max(1, median), position + 0.17),
                          xytext=(0, -3), textcoords="offset points", va="top", fontsize=8)
        ratio_summary[cohort] = {
            "count": len(rows), "min": float(raw.min()), "median": median, "max": float(raw.max()),
            "plotted_records": rows,
        }
    maximum = max(ratios["bridge"], key=lambda r: r["ratio"])
    outlier = (maximum["display_ratio"], maximum["jitter"] * 0.55)
    ratio_ax.plot(*outlier, marker="o", markersize=4.2, markerfacecolor="white",
                  markeredgecolor=INK, markeredgewidth=0.9, zorder=5)
    ratio_ax.annotate(f"{maximum['case']} ({maximum['ratio']:.1f}×)", outlier,
                      xytext=(0, 7), textcoords="offset points", ha="right", va="bottom", fontsize=8)
    ratio_ax.set_xscale("log")
    ratio_ax.set(xlim=(0.76, max(500, max(r["max"] for r in ratio_summary.values()) * 1.25)),
                 ylim=(2.50, -0.52), yticks=range(len(COHORTS)),
                 yticklabels=[f"{LABELS[c]}\n({len(ratios[c])})" for c in COHORTS],
                 xlabel="Upper bound / capacity saving")
    ratio_ax.set_xticks([1, 10, 100], labels=["1×", "10×", "100×"])
    ratio_ax.xaxis.set_minor_locator(NullLocator())
    ratio_ax.tick_params(axis="y", length=0)
    ratio_ax.spines["left"].set_visible(False)
    ratio_ax.axvline(1, color=GRAY, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    ratio_ax.set_title("(b)  When savings remain", loc="left", y=1.18, fontsize=9, fontweight="bold")
    ratio_ax.text(0, 1.02, "1× = bound equals saving", transform=ratio_ax.transAxes,
                  fontsize=8, ha="left", va="bottom", color=GRAY)
    tidy(ratio_ax, axis="x")
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
        "jitter_rule": "Within each cohort, vertical offset is 0.55 times the saved deterministic jitter; the ratio coordinate is unchanged.",
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
