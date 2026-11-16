"""Check the equality-only critical-cut edge case and preserve cohort cuts.

These checks target a concrete numerical defect: a zero plateau in the
max-flow residual previously led to a nonminimal reservation threshold,
while selecting strict slots only could omit a valid nonempty tight cut.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from critical_cut_audit import BASE, HERE, reservation_cuts


def main():
    edge_cases = [
        (
            "single_saturated_slot",
            {
                "N": 1,
                "T": 1,
                "availability": [[1]],
                "Pc": [1.0],
                "Etarget": [1.0],
                "E0": [0.0],
                "eta_c": [1.0],
                "dt": 1.0,
                "L": [0.0],
            },
            1.0,
            (0,),
        ),
        (
            "saturated_slot_with_idle_time",
            {
                "N": 1,
                "T": 3,
                "availability": [[0, 1, 0]],
                "Pc": [2.0],
                "Etarget": [2.0],
                "E0": [0.0],
                "eta_c": [1.0],
                "dt": 1.0,
                "L": [0.0, 0.0, 0.0],
            },
            2.0,
            (1,),
        ),
    ]
    results = []
    for name, case, expected_value, expected_cut in edge_cases:
        value, cuts = reservation_cuts(case, upper_coefficients=1.0)
        detected = {tuple(int(t) for t in cut["slots"]) for cut in cuts}
        if abs(value - expected_value) > 1e-9 or expected_cut not in detected:
            raise AssertionError((name, value, detected))
        results.append(
            {
                "name": name,
                "capacity_kw": value,
                "expected_capacity_kw": expected_value,
                "critical_slots_1based": [int(t) + 1 for t in expected_cut],
            }
        )

    with (HERE / "critical_cut_summary.csv").open() as stream:
        retained = {r["fleet"]: r for r in csv.DictReader(stream) if r["q"] == "0"}
    max_error = 0.0
    for path in sorted((BASE / "withdrawal_budget_cases").glob("*.json")):
        case = json.loads(path.read_text())
        value, cuts = reservation_cuts(case)
        row = retained[path.stem]
        expected = tuple(int(t) - 1 for t in row["slots_1based"].split())
        detected = {tuple(int(t) for t in cut["slots"]) for cut in cuts}
        error = abs(value - float(row["M_reservation_kw"]))
        max_error = max(max_error, error)
        if expected not in detected or error > 1e-7:
            raise AssertionError((path.stem, expected, detected, error))
    payload = {
        "edge_cases": results,
        "retained_cohort_cases": len(retained),
        "max_cohort_capacity_error_kw": max_error,
        "status": "passed",
    }
    (HERE / "critical_cut_edge_checks.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
