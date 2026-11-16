"""Independent small LP checks, explicitly enumerating withdrawal subsets.

Capacity only; exactness follows from the established cycle-removal argument.
Reconstructs all prefix energy and common-guard constraints directly with SciPy.
The purpose is structural validation, not solver-speed benchmarking.
"""

from __future__ import annotations

import csv
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from critical_cut_audit import BASE, HERE, reservation_cuts, residual_donor_credit
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cplex_lp import linprog
from scipy.sparse import csr_matrix


def solve_capacity(case, q, endpoints=(1.0, 1.5)):
    a = np.asarray(case["availability"], float)
    n, nt = a.shape
    nz = n * nt
    nv = 2 * nz + 1
    dt = float(case["dt"])
    ec, ed = np.asarray(case["eta_c"]), np.asarray(case["eta_d"])
    rows, rhs = [], []
    for i in range(n):
        for t in range(nt):
            r = np.zeros(nv)
            r[i * nt : i * nt + t + 1] = dt * ec[i]
            r[nz + i * nt : nz + i * nt + t + 1] = -dt / ed[i]
            rows.extend([r, -r])
            rhs.extend([case["Emax"][i] - case["E0"][i], case["E0"][i] - case["Emin"][i]])
            guard = r.copy()
            guard[nz:] = 0
            rows.append(guard)
            rhs.append(case["Emax"][i] - case["E0"][i])
        rows.append(-r)
        rhs.append(case["E0"][i] - case["Etarget"][i])
    energy_row_count = len(rows)
    for w in combinations(range(n), q):
        retained = np.ones(n)
        retained[list(w)] = 0
        for t in range(nt):
            for u in endpoints:
                r = np.zeros(nv)
                r[np.arange(n) * nt + t] = u
                r[nz + np.arange(n) * nt + t] = -u * retained
                r[-1] = -1
                rows.append(r)
                rhs.append(-case["L"][t])
    matrix = csr_matrix(np.asarray(rows))
    bounds = [(0, float(v)) for v in (a * np.asarray(case["Pc"])[:, None]).ravel()]
    bounds += [(0, float(v)) for v in (a * np.asarray(case["Pd"])[:, None]).ravel()]
    bounds += [(0, None)]
    obj = np.zeros(nv)
    obj[-1] = 1
    ans = linprog(obj, A_ub=matrix, b_ub=rhs, bounds=bounds)
    if not ans.success:
        raise RuntimeError((case.get("name"), q, ans.message))
    charge = ans.x[:nz].reshape(n, nt).copy()
    discharge = ans.x[nz : 2 * nz].reshape(n, nt).copy()
    k = (ec * ed)[:, None]
    removed = np.minimum(charge, discharge / k)
    charge -= removed
    discharge -= k * removed
    recovered = np.r_[charge.ravel(), discharge.ravel(), ans.fun]
    row_residuals = matrix @ recovered - np.asarray(rhs)
    energy_violation = float(max(0, np.max(row_residuals[:energy_row_count])))
    power_violation = float(max(0, np.max(row_residuals[energy_row_count:])))
    for j, (low, high) in enumerate(bounds):
        power_violation = max(power_violation, low - recovered[j])
        if high is not None:
            power_violation = max(power_violation, recovered[j] - high)
    overlap = float(np.max(np.minimum(charge, discharge)))
    if energy_violation > 1e-7 or power_violation > 1e-7 or overlap > 1e-7:
        raise AssertionError((q, energy_violation, power_violation, overlap))
    return {
        "M_kw": float(ans.fun),
        "recovered_power_violation_kw": power_violation,
        "recovered_energy_violation_kwh": energy_violation,
        # Legacy field combines units and is retained for compatibility.
        "recovered_max_violation": max(power_violation, energy_violation),
        "recovered_overlap_kw": overlap,
    }


def main():
    start = time.time()
    with (BASE / "withdrawal_budget_capacity_contexts.csv").open() as f:
        old = {r["fleet"]: r for r in csv.DictReader(f)}
    with (HERE / "critical_cut_summary.csv").open() as f:
        cert = {r["fleet"]: int(r["cut_collapse_q"]) for r in csv.DictReader(f) if r["q"] == "0"}
    records = []
    for path in sorted((BASE / "withdrawal_budget_cases").glob("*.json")):
        case = json.loads(path.read_text())
        fleet = path.stem
        for q in range(case["N"] + 1):
            result = solve_capacity(case, q)
            row = {
                "fleet": fleet,
                "q": q,
                **result,
                "saving_vs_reservation_kw": float(old[fleet]["M6_kw"]) - result["M_kw"],
                "cut_collapse_q": cert[fleet],
                "stored_value_error_kw": abs(result["M_kw"] - float(old[fleet][f"M{q}_kw"]))
                if q in (0, 1, 2, 6)
                else None,
            }
            if row["stored_value_error_kw"] is not None and row["stored_value_error_kw"] > 1e-7:
                raise AssertionError(row)
            if q >= cert[fleet] and abs(row["saving_vs_reservation_kw"]) > 1e-7:
                raise AssertionError(row)
            records.append(row)
    with (HERE / "full_frontier_independent_lp.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    # Empty donors cannot discharge before the first-slot deadline.
    counter = dict(
        name="prefix_SOC_counterexample",
        N=3,
        T=2,
        L=[0.0, 0.0],
        availability=[[1, 0], [1, 1], [1, 1]],
        E0=[0.0, 0.0, 0.0],
        Emin=[0.0, 0.0, 0.0],
        Emax=[3.0, 3.0, 3.0],
        Etarget=[1.8, 0.9, 0.9],
        Pc=[2.0, 2.0, 2.0],
        Pd=[1.0, 1.0, 1.0],
        eta_c=[0.9] * 3,
        eta_d=[0.9] * 3,
        dt=1.0,
    )
    counter_results = {q: solve_capacity(counter, q, (1.0,)) for q in range(4)}
    relaxed_credit = residual_donor_credit(np.ones((2, 1)), np.array([0.81, 0.81]), 1)
    if abs(relaxed_credit - 0.81) > 1e-8 or any(
        abs(r["M_kw"] - 2) > 1e-8 for r in counter_results.values()
    ):
        raise AssertionError((counter_results, relaxed_credit))
    (HERE / "prefix_soc_counterexample.json").write_text(
        json.dumps(
            {
                "case": counter,
                "q_results": counter_results,
                "critical_slots_1based": [1],
                "q1_relaxed_residual_donor_credit": relaxed_credit,
                "claim": "Positive Rq does not imply positive capacity savings; prefix SOC can prevent discharge.",
            },
            indent=2,
        )
        + "\n"
    )
    thresholds = []
    for fleet in old:
        rs = [r for r in records if r["fleet"] == fleet]
        observed = min(r["q"] for r in rs if r["saving_vs_reservation_kw"] <= 1e-7)
        thresholds.append(
            {
                "fleet": fleet,
                "observed_first_zero_q": observed,
                "sufficient_cut_collapse_q": cert[fleet],
            }
        )
    payload = {
        "cohort_lp_solves": len(records),
        "negative_control_lp_solves": len(counter_results),
        "new_cohort_q_levels": [3, 4, 5],
        "stored_values_compared": 96,
        "max_stored_difference_kw": max(r["stored_value_error_kw"] or 0 for r in records),
        "max_recovered_power_violation_kw": max(r["recovered_power_violation_kw"] for r in records),
        "max_recovered_energy_violation_kwh": max(
            r["recovered_energy_violation_kwh"] for r in records
        ),
        "max_recovered_row_violation": max(r["recovered_max_violation"] for r in records),
        "legacy_residual_note": "recovered_max_violation/max_recovered_row_violation are retained for backwards compatibility only and mix kW and kWh; use the separate power and energy fields for reporting",
        "false_zero_certificates": sum(
            r["q"] >= r["cut_collapse_q"] and r["saving_vs_reservation_kw"] > 1e-7 for r in records
        ),
        "thresholds": thresholds,
        "elapsed_seconds_not_a_benchmark": time.time() - start,
        "status": "Exploratory retrospective verification; all q now solved. Not an independent held-out sample.",
    }
    (HERE / "full_lp_check_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
