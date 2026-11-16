"""Independent scenario LP for the stated two-slot recharge-cancellation policy.

This is a restricted policy comparison, NOT an unrestricted recourse model.
The input grid is fixed in validation_protocol.json before execution.
"""
from __future__ import annotations

import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cplex_lp import linprog

from weighted_capacity import solve_capacity

HERE = Path(__file__).resolve().parent


def recourse_lp(n, q, D, b, ec, ed, H):
    k = ec * ed
    rows, rhs, scenarios = [], [], []
    for count in range(min(q, n) + 1):
        for w in combinations(range(n), count):
            keep = np.ones(n); keep[list(w)] = 0
            rows.extend([np.r_[-keep, -1.], np.r_[keep / k, -1.]])
            rhs.extend([-D, 0.]); scenarios.append((w, keep))
    ans = linprog(np.r_[np.zeros(n), 1.], A_ub=rows, b_ub=rhs,
                  bounds=[(0., b)] * n + [(0., None)])
    if not ans.success: raise RuntimeError(ans.message)
    y = ans.x[:n]
    error = 0.
    for _, keep in scenarios:
        discharge = keep * y
        charge = keep * y / k
        e0 = b / ed
        e1 = e0 - discharge / ed
        e2 = e1 + ec * charge
        error = max(error, float(max(0., -e1.min(), (e2 - (e0 + H)).max(),
                    np.max(np.abs(e2 - e0)), D - discharge.sum() - ans.fun,
                    charge.sum() - ans.fun)))
    assert error <= 1e-7
    return float(ans.fun), y.tolist(), error


def main():
    # Import only after definitions to avoid a circular import in protocol IO.
    from run_validation import verify_protocol
    verify_protocol()
    n, D, b, ec, ed = 4, 4., .75, .9, .9
    k = ec * ed
    rows, witnesses = [], []
    for H in (0., .25, .5, .75, 1.):
        for q in range(n + 1):
            model = {"name": f"contract_H{H}_q{q}", "N": n + 1, "T": 2,
                "availability": [[1., 0.]] + [[1., 1.]] * n, "duration": [1., 1.],
                "L": [0., 0.], "Ulo": [1., 1.], "Uhi": [1., 1.],
                "E0": [0.] + [b / ed] * n, "Etarget": [ec * D] + [b / ed] * n,
                "Emin": [0.] * (n + 1), "Emax": [ec * D] + [b / ed + H] * n,
                "Pc": [D] + [b / k] * n, "Pd": [0.] + [b] * n,
                "eta_c": [ec] * (n + 1), "eta_d": [ed] * (n + 1)}
            fixed, schedule = solve_capacity(model, q)
            rec, y, rec_error = recourse_lp(n, q, D, b, ec, ed, H)
            A = n * D / (n + k * (n - q))
            expected_fixed = max(D - (n - q) * min(b, ed * H), A) if q < n else D
            expected_rec = max(D - (n - q) * b, A) if q < n else D
            assert fixed["status"] == "optimal"
            assert max(abs(fixed["M_kw"] - expected_fixed), abs(rec - expected_rec)) <= 1e-7
            rows.append({"H_kwh": H, "q": q, "fixed_M_kw": fixed["M_kw"],
                         "restricted_recourse_M_kw": rec, "fixed_contract_cost_kw": fixed["M_kw"] - rec,
                         "formula_fixed_M_kw": expected_fixed, "formula_recourse_M_kw": expected_rec,
                         "fixed_energy_error_kwh": fixed["energy_residual_kwh"],
                         "fixed_power_error_kw": fixed["power_residual_kw"], "recourse_direct_error": rec_error})
            witnesses.append({"H_kwh": H, "q": q, "fixed": {key: val.tolist() for key, val in schedule.items()},
                              "restricted_recourse_precommitted_discharge_kw": y})
    destination = HERE / "results"
    destination.mkdir(exist_ok=True)
    with (destination / "contract_cost.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {"scope": "Specified two-slot cancel-own-recharge policy only; not unrestricted adaptive scheduling",
               "fixed_charging_LPs": len(rows), "restricted_scenario_LPs": len(rows),
               "maximum_formula_error_kw": max(max(abs(r["fixed_M_kw"] - r["formula_fixed_M_kw"]), abs(r["restricted_recourse_M_kw"] - r["formula_recourse_M_kw"])) for r in rows),
               "witnesses": witnesses}
    (destination / "contract_cost_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "witnesses"}))


if __name__ == "__main__":
    main()
