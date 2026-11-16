"""Predeclared, isolated check of the homogeneous two-slot capacity formula.

Uses the independently assembled full LP with explicit withdrawal subsets;
checks the proposed complementary witness separately. Not a timing benchmark.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
from full_lp_checks import solve_capacity

HERE = Path(__file__).resolve().parent
TOL = 1e-7


def grid():
    cases = []
    for n in (1, 2, 4):
        for eta in (0.8, 0.95):
            k = eta * eta
            for b in (0.1, 2.0):
                for q in range(n + 2):
                    cases.append(
                        dict(
                            n=n,
                            eta=eta,
                            k=k,
                            b=b,
                            q=q,
                            D=2.0,
                            L=3.0,
                            grid_part="fixed_power_budgets",
                        )
                    )
            for q in range(n):
                cases.append(
                    dict(
                        n=n,
                        eta=eta,
                        k=k,
                        b=k * 2 / (n + k * (n - q)),
                        q=q,
                        D=2.0,
                        L=3.0,
                        grid_part="branch_boundary",
                    )
                )
    assert len(cases) == 66
    return cases


def make_case(spec):
    n, eta, k, b, D, L = (spec[x] for x in ("n", "eta", "k", "b", "D", "L"))
    return dict(
        name="homogeneous_two_slot",
        N=n + 1,
        T=2,
        dt=1.0,
        L=[L, L],
        availability=[[1.0, 0.0]] + [[1.0, 1.0] for _ in range(n)],
        E0=[0.0] + [b / eta] * n,
        Emin=[0.0] * (n + 1),
        Etarget=[eta * D] + [b / eta] * n,
        Emax=[eta * D] + [2 * b / eta] * n,
        Pc=[D] + [b / k] * n,
        Pd=[0.0] + [b] * n,
        eta_c=[eta] * (n + 1),
        eta_d=[eta] * (n + 1),
    )


def formula_and_witness(spec):
    n, q, k, b, D, L = (spec[x] for x in ("n", "q", "k", "b", "D", "L"))
    if q >= n:
        y, m, branch = 0.0, L + D, "full_withdrawal"
    else:
        power_bound = D - (n - q) * b
        recharge_bound = n * D / (n + k * (n - q))
        y = min(b, k * D / (n + k * (n - q)))
        m = L + max(power_bound, recharge_bound)
        branch = (
            "boundary"
            if abs(power_bound - recharge_bound) <= 1e-12
            else "power_limited"
            if power_bound > recharge_bound
            else "recharge_limited"
        )
    c = np.zeros((n + 1, 2))
    d = np.zeros_like(c)
    c[0, 0] = D
    c[1:, 1] = y / k
    d[1:, 0] = y
    return m, y, branch, c, d


def check_witness(case, q, m, c, d):
    ec = np.asarray(case["eta_c"])[:, None]
    ed = np.asarray(case["eta_d"])[:, None]
    initial = np.asarray(case["E0"])[:, None]
    energy = initial + np.cumsum(ec * c - d / ed, axis=1)
    no_discharge_energy = initial + np.cumsum(ec * c, axis=1)
    availability = np.asarray(case["availability"])
    violations = {
        "charging_upper": float(np.max(c - availability * np.asarray(case["Pc"])[:, None])),
        "discharging_upper": float(np.max(d - availability * np.asarray(case["Pd"])[:, None])),
        "nonnegativity": float(max(np.max(-c), np.max(-d))),
        "energy_lower": float(np.max(np.asarray(case["Emin"])[:, None] - energy)),
        "energy_upper": float(np.max(energy - np.asarray(case["Emax"])[:, None])),
        "common_no_discharge_guard": float(
            np.max(no_discharge_energy - np.asarray(case["Emax"])[:, None])
        ),
        "departure_target": float(np.max(np.asarray(case["Etarget"]) - energy[:, -1])),
        "complementarity": float(np.max(np.minimum(c, d))),
    }
    worst_capacity = -float("inf")
    for subset in combinations(range(case["N"]), q):
        delivered = d.copy()
        if subset:
            delivered[list(subset)] = 0
        worst_capacity = max(
            worst_capacity,
            float(np.max(np.asarray(case["L"]) + c.sum(axis=0) - delivered.sum(axis=0))),
        )
    violations["withdrawal_capacity"] = worst_capacity - m
    return {
        "max_violation": max(0.0, *violations.values()),
        "violations": {key: max(0.0, value) for key, value in violations.items()},
        "worst_withdrawal_capacity_kw": worst_capacity,
        "witness_capacity_value_error_kw": abs(worst_capacity - m),
    }


def write_output(payload):
    temporary = HERE / "sharp_class_independent_check.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(HERE / "sharp_class_independent_check.json")


def main():
    specifications = grid()
    payload = {
        "status": "grid declared before any solve",
        "declared_utc": datetime.now(timezone.utc).isoformat(),
        "planned_lp_solves": len(specifications),
        "absolute_tolerance_kw": TOL,
        "coefficient_endpoints": [1.0],
        "model": "One recipient, n homogeneous donors; two slots; common no-discharge guard; explicit withdrawal subsets.",
        "formula": "q<n: L+max(D-(n-q)*b, n*D/(n+k*(n-q))); q>=n: L+D",
        "grid_specifications": specifications,
        "source_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("sharp_class_independent_check.py", "full_lp_checks.py")
        },
        "runtime_versions": {"numpy": np.__version__, "scipy": scipy.__version__},
    }
    write_output(payload)
    records = []
    for j, spec in enumerate(specifications):
        case = make_case(spec)
        expected, y, branch, c, d = formula_and_witness(spec)
        witness = check_witness(case, spec["q"], expected, c, d)
        result = solve_capacity(case, spec["q"], endpoints=(1.0,))
        error = abs(result["M_kw"] - expected)
        records.append(
            {
                "index": j,
                **spec,
                "regime": branch,
                "formula_M_kw": expected,
                "uniform_donor_discharge_kw": y,
                "lp_result": result,
                "formula_absolute_error_kw": error,
                "explicit_witness_check": witness,
                "passed": error <= TOL
                and witness["max_violation"] <= TOL
                and witness["witness_capacity_value_error_kw"] <= TOL,
            }
        )
    payload.update(
        status="completed",
        finished_utc=datetime.now(timezone.utc).isoformat(),
        completed_lp_solves=len(records),
        passed_count=sum(r["passed"] for r in records),
        failure_count=sum(not r["passed"] for r in records),
        max_formula_error_kw=max(r["formula_absolute_error_kw"] for r in records),
        max_lp_recovered_violation=max(r["lp_result"]["recovered_max_violation"] for r in records),
        max_lp_recovered_overlap_kw=max(r["lp_result"]["recovered_overlap_kw"] for r in records),
        max_explicit_witness_violation=max(
            r["explicit_witness_check"]["max_violation"] for r in records
        ),
        max_explicit_witness_value_error_kw=max(
            r["explicit_witness_check"]["witness_capacity_value_error_kw"] for r in records
        ),
        regimes={
            name: sum(r["regime"] == name for r in records)
            for name in ("power_limited", "recharge_limited", "boundary", "full_withdrawal")
        },
        records=records,
        interpretation="Finite numerical corroboration of the restricted-class formula and constructive witness; not a substitute for the proof, not a general-fleet exact characterization, and not a timing benchmark.",
    )
    write_output(payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "completed_lp_solves",
                    "passed_count",
                    "failure_count",
                    "max_formula_error_kw",
                    "max_lp_recovered_violation",
                    "max_lp_recovered_overlap_kw",
                    "max_explicit_witness_violation",
                    "max_explicit_witness_value_error_kw",
                    "regimes",
                )
            },
            indent=2,
        )
    )
    if payload["failure_count"]:
        raise AssertionError(
            "The predeclared sharp-class check has failed cases; inspect the saved JSON."
        )


if __name__ == "__main__":
    main()
