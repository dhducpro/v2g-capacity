"""Exploratory verification of a proposed critical-cut V2G capacity bound.

This does not alter the manuscript or any existing experiment. The 24 fleets
were previously selected; this is a retrospective structural analysis, not
independent validation. Reservation cuts use a standard bipartite max-flow
condition. The new candidate is the recharge-adjusted robust donor bound.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cplex_lp import linprog
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
BASE = HERE / "inputs"
if not BASE.exists():
    raise FileNotFoundError("The accompanying inputs/ directory is required.")
TOL = 1e-7


def reservation_cuts(case, upper_coefficients=1.5):
    """Find certified tight cuts by enumerating the small fleet's EV subsets.

    For J, sum_i H_i <= sum_t min(b_t(M), sum_i P_it), with b=(M-L)/u.
    This implementation deliberately requires positive U and M>=max L.
    The default upper coefficient is the frozen cohort value. A monotone
    threshold search returns the *first* feasible capacity, including when
    the max-flow gap is zero on a whole interval of capacities.
    """
    a = np.asarray(case["availability"], float)
    n, nt = a.shape
    pc = a * np.asarray(case["Pc"])[:, None]
    h = (np.asarray(case["Etarget"]) - case["E0"]) / (np.asarray(case["eta_c"]) * case["dt"])
    demand = np.maximum(h, 0)
    load = np.asarray(case["L"], float)
    u = np.broadcast_to(np.asarray(upper_coefficients, float), (nt,)).copy()
    if np.any(u <= 0):
        raise ValueError("This cut extractor requires positive upper coefficients")
    lo = max(0.0, float(load.max()))
    hi = float(np.max(load + u * pc.sum(axis=0))) + 1
    subsets = []
    optimum = lo
    for mask in range(1, 1 << n):
        idx = [i for i in range(n) if mask & (1 << i)]
        cap = pc[idx].sum(axis=0)
        need = float(demand[idx].sum())

        def gap(m):
            return float(np.minimum((m - load) / u, cap).sum() - need)

        if gap(hi) < -TOL:
            raise ValueError("Physically infeasible reservation fleet")
        if gap(lo) >= 0:
            threshold = lo
        else:
            # A saturated subset can have a zero plateau. Use its left endpoint.
            left, right = lo, hi
            for _ in range(80):
                mid = (left + right) / 2
                if gap(mid) >= 0:
                    right = mid
                else:
                    left = mid
            threshold = right
        optimum = max(optimum, threshold)
        subsets.append((mask, threshold, cap))
    if optimum <= lo + TOL:
        return optimum, []
    cuts = {}
    for mask, threshold, cap in subsets:
        if abs(threshold - optimum) > TOL:
            continue
        b = (optimum - load) / u
        # When all tight slots are ties, keep them so the critical cut stays nonempty.
        s = np.flatnonzero(cap > b + 1e-9)
        if not len(s):
            s = np.flatnonzero(np.abs(cap - b) <= 1e-9)
        if not len(s):
            raise AssertionError((mask, "Binding subset has no candidate time cut"))
        outside = pc.sum(axis=1) - pc[:, s].sum(axis=1)
        required = float(np.maximum(h - outside, 0).sum())
        implied = float((required + np.sum(load[s] / u[s])) / np.sum(1 / u[s]))
        if abs(implied - optimum) > TOL:
            raise AssertionError((mask, implied, optimum))
        cuts[tuple(int(t) for t in s)] = {
            "slots": s,
            "outside": outside,
            "h": h,
            "required": required,
            "capacity": implied,
            "denominator": float(np.sum(1 / u[s])),
        }
    return optimum, list(cuts.values())


def residual_donor_credit(pd, energy, q):
    """Maximize sum_t(sum z-top_q(z)) with donor power/energy budgets."""
    n, nt = pd.shape
    if q == 0:
        return float(np.minimum(pd.sum(axis=1), energy).sum())
    if q >= n:
        return 0.0
    nz = n * nt
    theta0, r0 = nz, nz + nt
    nv = r0 + nz
    objective = np.zeros(nv)
    objective[:nz] = -1
    objective[theta0:r0] = q
    objective[r0:] = 1
    rows, cols, vals, rhs = [], [], [], []
    row = 0
    for i in range(n):
        for t in range(nt):
            rows.append(row)
            cols.append(i * nt + t)
            vals.append(1)
        rhs.append(float(energy[i]))
        row += 1
    for i in range(n):
        for t in range(nt):
            for col, val in ((i * nt + t, 1), (theta0 + t, -1), (r0 + i * nt + t, -1)):
                rows.append(row)
                cols.append(col)
                vals.append(val)
            rhs.append(0.0)
            row += 1
    matrix = coo_matrix((vals, (rows, cols)), shape=(row, nv)).tocsr()
    bounds = [(0, float(x)) for x in pd.ravel()] + [(0, None)] * (nv - nz)
    result = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=bounds)
    if not result.success:
        raise RuntimeError(result.message)
    return max(0.0, -float(result.fun))


def main():
    with (BASE / "withdrawal_budget_capacity_contexts.csv").open() as f:
        observed = {r["fleet"]: r for r in csv.DictReader(f)}
    detailed, summary, provenance = [], [], []
    for path in sorted((BASE / "withdrawal_budget_cases").glob("*.json")):
        case = json.loads(path.read_text())
        fleet = path.stem
        a = np.asarray(case["availability"], float)
        k = np.asarray(case["eta_c"]) * case["eta_d"]
        guard_budget = np.maximum(
            0, np.asarray(case["eta_d"]) * (np.asarray(case["Emax"]) - case["Etarget"]) / case["dt"]
        )
        baseline, cuts = reservation_cuts(case)
        baseline_error = abs(baseline - float(observed[fleet]["M6_kw"]))
        if baseline_error > TOL:
            raise AssertionError((fleet, baseline, observed[fleet]["M6_kw"]))
        if not cuts:
            raise ValueError((fleet, "No charging-driven tight cut"))
        for q in (0, 1, 2, 6):
            candidates = []
            for cut_id, cut in enumerate(cuts):
                s = cut["slots"]
                slack = np.maximum(cut["outside"] - cut["h"], 0)
                useful = np.minimum(guard_budget, k * slack)
                useful[useful < 1e-9] = 0
                pd = a[:, s] * np.asarray(case["Pd"])[:, None]
                donor_count = ((pd > 1e-9) & (useful[:, None] > 1e-9)).sum(axis=0)
                collapse_q = int(donor_count.max())
                residual = residual_donor_credit(pd, useful, q)
                bound = residual / cut["denominator"]
                true_saving = baseline - float(observed[fleet][f"M{q}_kw"])
                if true_saving > bound + TOL:
                    raise AssertionError((fleet, q, true_saving, bound))
                if (residual <= TOL) != (q >= collapse_q):
                    raise AssertionError("LP credit and donor-zero criterion disagree")
                row = {
                    "fleet": fleet,
                    "q": q,
                    "cut_id": cut_id,
                    "slots_1based": " ".join(str(int(t) + 1) for t in s),
                    "donors_positive_budget_1based": " ".join(
                        str(i + 1) for i in range(case["N"]) if useful[i] > 1e-9
                    ),
                    "M_reservation_kw": baseline,
                    "observed_saving_kw": true_saving,
                    "upper_bound_saving_kw": bound,
                    "cut_collapse_q": collapse_q,
                    "certifies_zero_saving": q >= collapse_q,
                    "baseline_error_kw": baseline_error,
                }
                detailed.append(row)
                candidates.append(row)
            best = min(candidates, key=lambda r: r["upper_bound_saving_kw"])
            summary.append(best)
        provenance.append(
            {"fleet": fleet, "case_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    for name, data in (
        ("critical_cut_details.csv", detailed),
        ("critical_cut_summary.csv", summary),
    ):
        with (HERE / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    totals = {}
    for q in (0, 1, 2, 6):
        rs = [r for r in summary if r["q"] == q]
        totals[q] = {
            "fleets": len(rs),
            "observed_zero_saving": sum(r["observed_saving_kw"] <= TOL for r in rs),
            "certified_zero_saving": sum(r["certifies_zero_saving"] for r in rs),
            "false_zero_certificates": sum(
                r["certifies_zero_saving"] and r["observed_saving_kw"] > TOL for r in rs
            ),
            "max_baseline_error_kw": max(r["baseline_error_kw"] for r in rs),
            "max_bound_violation_kw": max(
                0, max(r["observed_saving_kw"] - r["upper_bound_saving_kw"] for r in rs)
            ),
        }
    payload = {
        "status": "retrospective exploratory verification; not held-out validation",
        "scope": "current common guard, fixed sessions, constant [1,1.5] coefficients; no economic optimality claim",
        "reservation_cut_method": "standard max-flow EV-subset cuts; minimum feasible threshold including zero plateaus; strict slots with equality-slot fallback when empty; verified detected tight cuts, no exhaustive minimal-donor assertion",
        "totals": totals,
        "input_provenance": provenance,
    }
    (HERE / "critical_cut_audit_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(totals, indent=2))
    for r in summary:
        if r["q"] in (1, 2):
            print(
                r["fleet"],
                r["q"],
                "gap",
                round(r["observed_saving_kw"], 6),
                "bound",
                round(r["upper_bound_saving_kw"], 6),
                "collapse_q",
                r["cut_collapse_q"],
                "cut",
                r["slots_1based"],
            )


if __name__ == "__main__":
    main()
