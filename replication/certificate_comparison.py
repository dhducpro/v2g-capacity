"""Compare the retained cut certificate with the conditional occupancy bound.

This is a deterministic audit of stored inputs and capacity values. It runs
no optimization and does not select another fleet or revise observed results.
The bound-to-saving ratio describes positive observations at q=0,1,2 only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "replication/capacity_checks"
OUT = ROOT / "outputs"
TOL = 1e-6


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    with (CHECKS / "critical_cut_summary.csv").open() as stream:
        cut_rows = list(csv.DictReader(stream))
    with (CHECKS / "full_frontier_independent_lp.csv").open() as stream:
        frontier = list(csv.DictReader(stream))
    cut_by_fleet = {r["fleet"]: r for r in cut_rows if r["q"] == "0"}
    comparison = []
    for path in sorted((CHECKS / "inputs/withdrawal_budget_cases").glob("*.json")):
        case = json.loads(path.read_text())
        fleet = path.stem
        a_max = int(np.asarray(case["availability"]).sum(axis=0).max())
        cut = cut_by_fleet[fleet]
        reservation = float(cut["M_reservation_kw"])
        baseline_load = float(max(case["L"]))
        if not reservation > baseline_load + TOL:
            raise AssertionError((fleet, "Conditional occupancy bound not applicable"))
        cut_q = int(cut["cut_collapse_q"])
        zero_q = min(
            int(r["q"])
            for r in frontier
            if r["fleet"] == fleet and float(r["saving_vs_reservation_kw"]) <= TOL
        )
        comparison.append(
            {
                "fleet": fleet,
                "maximum_occupancy": a_max,
                "occupancy_sufficient_q": a_max - 1,
                "retained_cut_sufficient_q": cut_q,
                "observed_first_zero_q": zero_q,
                "threshold_improvement": a_max - 1 - cut_q,
                "reservation_capacity_kw": reservation,
                "maximum_background_kw": baseline_load,
                "critical_slots_1based": cut["slots_1based"],
                "case_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    ratios = [
        {
            "fleet": r["fleet"],
            "q": int(r["q"]),
            "saving_kw": float(r["observed_saving_kw"]),
            "cut_bound_kw": float(r["upper_bound_saving_kw"]),
            "bound_to_saving_ratio": float(r["upper_bound_saving_kw"])
            / float(r["observed_saving_kw"]),
        }
        for r in cut_rows
        if int(r["q"]) in (0, 1, 2) and float(r["observed_saving_kw"]) > TOL
    ]
    vals = np.asarray([r["bound_to_saving_ratio"] for r in ratios])
    q1 = [r for r in frontier if int(r["q"]) == 1]
    q1_total = sum(max(0.0, float(r["saving_vs_reservation_kw"])) for r in q1)
    q1_w206 = next(float(r["saving_vs_reservation_kw"]) for r in q1 if r["fleet"] == "W206")
    payload = {
        "fleet_count": len(comparison),
        "strictly_smaller_cut_threshold_count": sum(
            r["threshold_improvement"] > 0 for r in comparison
        ),
        "equal_threshold_fleets": [
            r["fleet"] for r in comparison if r["threshold_improvement"] == 0
        ],
        "threshold_improvement_min": min(r["threshold_improvement"] for r in comparison),
        "threshold_improvement_max": max(r["threshold_improvement"] for r in comparison),
        "cut_threshold_matches_observed_count": sum(
            r["retained_cut_sufficient_q"] == r["observed_first_zero_q"] for r in comparison
        ),
        "positive_bound_ratio_observation_count": len(ratios),
        "positive_bound_ratio_q_levels": [0, 1, 2],
        "positive_bound_ratio_min": float(vals.min()),
        "positive_bound_ratio_median": float(np.median(vals)),
        "positive_bound_ratio_max": float(vals.max()),
        "largest_ratio_observation": max(ratios, key=lambda r: r["bound_to_saving_ratio"]),
        "W206_share_of_total_q1_saving": q1_w206 / q1_total,
        "interpretation": "Retrospective comparison on the same 24 constructed fleets. The retained cut improves the sufficient occupancy threshold, but is not claimed to be the best of all critical cuts. Positive bounds need not be tight or attainable; observations across q are not independent fleets.",
        "tolerance_kw": TOL,
    }
    write_csv(OUT / "certificate_threshold_comparison.csv", comparison)
    write_csv(OUT / "certificate_positive_bound_ratios.csv", ratios)
    (OUT / "certificate_comparison_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        r"\begin{tabular}{lrrr@{\qquad}lrrr}",
        r"\toprule",
        r"Fleet & $q_{\rm occ}$ & $\widehat q_{\rm cut}$ & $q_{\rm zero}$ & Fleet & $q_{\rm occ}$ & $\widehat q_{\rm cut}$ & $q_{\rm zero}$\\",
        r"\midrule",
    ]
    for left, right in zip(comparison[:12], comparison[12:]):
        parts = []
        for row in (left, right):
            parts += [
                row["fleet"],
                str(row["occupancy_sufficient_q"]),
                str(row["retained_cut_sufficient_q"]),
                str(row["observed_first_zero_q"]),
            ]
        lines.append(" & ".join(parts) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "certificate_comparison_table.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
