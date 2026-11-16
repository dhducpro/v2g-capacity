"""Small mathematical implementation checks, not empirical validation fleets.

Checks use analytic two-slot values, exact exhaustive time-subset cuts, an
independent prefix-energy/withdrawal-enumeration LP, and interval splitting.
These fixtures are not counted as held-out experimental observations.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent


def enumerated_capacity(m, q):
    """Independent dense prefix-energy formulation and explicit withdrawals."""
    a = np.asarray(m["availability"], float)
    n, nt = a.shape
    z, nv = n * nt, 2 * n * nt + 1
    rows, limits = [], []
    dt = np.asarray(m["duration"])
    for i in range(n):
        for t in range(nt):
            row = np.zeros(nv)
            row[i * nt : i * nt + t + 1] = dt[: t + 1] * m["eta_c"][i]
            row[z + i * nt : z + i * nt + t + 1] = -dt[: t + 1] / m["eta_d"][i]
            rows.extend([row.copy(), -row.copy()])
            limits.extend([m["Emax"][i] - m["E0"][i], m["E0"][i] - m["Emin"][i]])
            guard = row.copy()
            guard[z:] = 0
            rows.append(guard)
            limits.append(m["Emax"][i] - m["E0"][i])
        rows.append(-row)
        limits.append(m["E0"][i] - m["Etarget"][i])
    for lost in itertools.combinations(range(n), q):
        for t in range(nt):
            for u in {float(m["Ulo"][t]), float(m["Uhi"][t])}:
                row = np.zeros(nv)
                row[np.arange(n) * nt + t] = u
                row[z + np.arange(n) * nt + t] = -u
                row[z + np.asarray(lost, dtype=int) * nt + t] = 0
                row[-1] = -1
                rows.append(row)
                limits.append(-m["L"][t])
    bounds = [(0, v) for v in (a * np.asarray(m["Pc"])[:, None]).ravel()]
    bounds += [(0, v) for v in (a * np.asarray(m["Pd"])[:, None]).ravel()]
    bounds.append((0, None))
    objective = np.zeros(nv)
    objective[-1] = 1
    result = linprog(objective, A_ub=np.asarray(rows), b_ub=limits, bounds=bounds, method="highs")
    if not result.success:
        raise AssertionError(result.message)
    return float(result.fun)


def all_time_cut_value(m):
    a, dt = np.asarray(m["availability"]), np.asarray(m["duration"])
    h = (np.asarray(m["Etarget"]) - m["E0"]) / m["eta_c"]
    pc = a * np.asarray(m["Pc"])[:, None] * dt
    eligible = np.flatnonzero(np.asarray(m["Uhi"]) > 0)
    optimum = max(0.0, max(m["L"]))
    for mask in range(1, 1 << len(eligible)):
        s = eligible[[bool(mask & (1 << j)) for j in range(len(eligible))]]
        outside = pc.sum(axis=1) - pc[:, s].sum(axis=1)
        phi = np.maximum(h - outside, 0).sum()
        ratio = (phi + np.sum(dt[s] * np.asarray(m["L"])[s] / np.asarray(m["Uhi"])[s])) / np.sum(
            dt[s] / np.asarray(m["Uhi"])[s]
        )
        optimum = max(optimum, float(ratio))
    return optimum


def analytic_family(n=3, demand=4.0, rating=0.7, q=0):
    eta = 0.9
    k = eta * eta
    m = {
        "name": "analytic_two_slot",
        "N": n + 1,
        "T": 2,
        "availability": np.array([[1, 0]] + [[1, 1]] * n),
        "duration": np.ones(2),
        "L": np.zeros(2),
        "Ulo": np.ones(2),
        "Uhi": np.ones(2),
        "E0": np.array([0.0] + [rating / eta] * n),
        "Emin": np.zeros(n + 1),
        "Etarget": np.array([eta * demand] + [rating / eta] * n),
        "Emax": np.array([eta * demand] + [2 * rating / eta] * n),
        "Pc": np.array([demand] + [rating / k] * n),
        "Pd": np.array([0.0] + [rating] * n),
        "eta_c": np.full(n + 1, eta),
        "eta_d": np.full(n + 1, eta),
    }
    expected = demand if q >= n else max(demand - (n - q) * rating, n * demand / (n + k * (n - q)))
    return m, expected


def fixtures():
    yield analytic_family()[0]
    yield {
        "name": "unequal_signed_heterogeneous",
        "availability": np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0]]),
        "duration": np.array([0.3, 1.2, 0.7]),
        "L": np.array([1.0, 0.4, 1.3]),
        "Ulo": np.array([0.8, 0.2, 1.0]),
        "Uhi": np.array([1.2, 0.9, 1.5]),
        "E0": np.array([0.0, 2.0, 0.4, 0.9]),
        "Emin": np.zeros(4),
        "Etarget": np.array([0.81, 1.5, 1.3, 1.1]),
        "Emax": np.array([0.81, 3.0, 3.0, 2.0]),
        "Pc": np.array([3.0, 2.0, 1.7, 1.2]),
        "Pd": np.array([0.0, 1.3, 0.8, 0.7]),
        "eta_c": np.array([0.9, 0.95, 0.85, 0.92]),
        "eta_d": np.array([0.93, 0.85, 0.91, 0.88]),
    }
    yield {
        "name": "one_slot_equality_cut",
        "availability": np.ones((1, 1)),
        "duration": np.array([0.37]),
        "L": np.array([1.2]),
        "Ulo": np.array([0.8]),
        "Uhi": np.array([1.3]),
        "E0": np.array([0.0]),
        "Emin": np.array([0.0]),
        "Emax": np.array([0.9 * 2 * 0.37]),
        "Etarget": np.array([0.9 * 2 * 0.37]),
        "Pc": np.array([2.0]),
        "Pd": np.array([1.0]),
        "eta_c": np.array([0.9]),
        "eta_d": np.array([0.8]),
    }
    zero = analytic_family(n=2)[0]
    zero = {
        **zero,
        "name": "zero_coefficient_second_slot",
        "Ulo": np.array([1.0, 0.0]),
        "Uhi": np.array([1.0, 0.0]),
    }
    yield zero
    yield {
        "name": "background_surplus",
        "availability": np.ones((1, 2)),
        "duration": np.array([0.4, 0.6]),
        "L": np.array([2.0, 2.0]),
        "Ulo": np.ones(2),
        "Uhi": np.ones(2),
        "E0": np.array([1.0]),
        "Emin": np.array([0.0]),
        "Emax": np.array([1.5]),
        "Etarget": np.array([0.0]),
        "Pc": np.array([1.0]),
        "Pd": np.array([1.0]),
        "eta_c": np.array([1.0]),
        "eta_d": np.array([1.0]),
    }


def run(solver=None, report_path=None):
    if solver is None:
        import weighted_capacity as solver

        if report_path is None:
            report_path = HERE / "weighted_implementation_checks.json"
    checks = []
    for m in fixtures():
        n, nt = np.asarray(m["availability"]).shape
        reserve, _ = solver.solve_reservation(m)
        allcut = all_time_cut_value(m)
        assert abs(reserve["M_kw"] - allcut) < 1e-7, (m["name"], reserve, allcut)
        extracted = solver.reservation_cut(m, reserve["M_kw"])
        if reserve["M_kw"] > max(0, max(m["L"])) + 1e-6:
            assert extracted["status"] == "verified_critical_cut", (m["name"], extracted)
        else:
            assert extracted["status"] == "background_dominated_no_certificate"
        for q in range(n + 1):
            result, schedule = solver.solve_capacity(m, q)
            independent = enumerated_capacity(m, q)
            assert result["status"] == "optimal"
            assert abs(result["M_kw"] - independent) < 1e-7, (m["name"], q, result, independent)
            if q == n:
                assert abs(result["M_kw"] - reserve["M_kw"]) < 1e-7
            for cut in extracted["cuts"]:
                cert = solver.compute_certificate(m, cut, q)
                assert reserve["M_kw"] - result["M_kw"] <= cert["bound_kw"] + 1e-7, (
                    m["name"],
                    q,
                    cert,
                    result,
                )
            checks.append(
                {
                    "fixture": m["name"],
                    "q": q,
                    "M_kw": result["M_kw"],
                    "enumeration_error_kw": abs(result["M_kw"] - independent),
                    "reservation_cut_error_kw": abs(reserve["M_kw"] - allcut),
                    "cut_status": extracted["status"],
                    **solver.audit_schedule(
                        m, q, schedule["charge_kw"], schedule["discharge_kw"], result["M_kw"]
                    ),
                }
            )
            split = {
                **m,
                "name": m["name"] + "_split",
                "T": 2 * nt,
                "availability": np.repeat(m["availability"], 2, axis=1),
                "duration": np.repeat(m["duration"], 2) / 2,
                **{k: np.repeat(m[k], 2) for k in ("L", "Ulo", "Uhi")},
            }
            divided, _ = solver.solve_capacity(split, q)
            assert abs(divided["M_kw"] - result["M_kw"]) < 1e-7, (
                m["name"],
                q,
                "split",
                divided,
                result,
            )
    analytical = []
    for n in (1, 3):
        for rating in (0.2, 1.5):
            for q in range(n + 2):
                m, expected = analytic_family(n=n, rating=rating, q=q)
                result, _ = solver.solve_capacity(m, q)
                assert abs(result["M_kw"] - expected) < 1e-7, (n, rating, q, result, expected)
                analytical.append(
                    {
                        "n": n,
                        "rating": rating,
                        "q": q,
                        "absolute_error_kw": abs(result["M_kw"] - expected),
                    }
                )
    infeasible = list(fixtures())[2]
    infeasible = {**infeasible, "name": "infeasible_full_power_request", "Pc": np.array([1.0])}
    infeasible_result, infeasible_schedule = solver.solve_capacity(infeasible, 0)
    infeasible_reservation = solver.reservation_cut(infeasible)
    assert infeasible_result["status"] == "infeasible" and infeasible_schedule is None
    assert infeasible_reservation["status"] == "infeasible_reservation"
    # A tiny interval can turn a small energy allowance into appreciable power.
    tiny = {
        "name": "tiny_energy_large_power_credit",
        "availability": np.ones((3, 1)),
        "duration": np.array([1e-8]),
        "L": np.zeros(1),
        "Ulo": np.ones(1),
        "Uhi": np.ones(1),
        "E0": np.array([0.0, 5e-10, 5e-10]),
        "Etarget": np.array([1e-8, 0.0, 0.0]),
        "Emin": np.zeros(3),
        "Emax": np.array([1e-8, 5e-10, 5e-10]),
        "Pc": np.array([1.0, 0.0, 0.0]),
        "Pd": np.array([0.0, 1.0, 1.0]),
        "eta_c": np.ones(3),
        "eta_d": np.ones(3),
    }
    tiny_cut = solver._cut_data(solver._model(tiny), [0], 1.0)
    tiny_bound = solver.compute_certificate(tiny, tiny_cut, 1)
    assert tiny_cut["q_certificate"] == 2
    assert tiny_bound["bound_kw"] >= 0.05 - 1e-12 and not tiny_bound["certifies_zero_saving"]
    slack_cut = solver._cut_data(solver._model(tiny), [0], 1.0 + 5e-7)
    slack_bound = solver.compute_certificate(tiny, slack_cut, 3)
    assert 4.99e-7 < slack_bound["bound_kw"] < 5.01e-7
    finite_check = False
    try:
        solver.audit_schedule(tiny, 0, np.full((3, 1), np.nan), np.zeros((3, 1)), 1.0)
    except ValueError:
        finite_check = True
    assert finite_check
    output = {
        "status": "PASS",
        "scope": "Implementation fixtures only; not empirical validation observations",
        "exhaustive_withdrawal_and_split_checks": checks,
        "analytical_two_slot_checks": analytical,
        "fixture_count": len(set(r["fixture"] for r in checks)),
        "infeasible_status_retained": True,
        "tiny_interval_certificate_check": tiny_bound,
        "near_critical_signed_slack_check": slack_bound,
        "nonfinite_schedule_rejected": finite_check,
        "max_enumeration_error_kw": max(r["enumeration_error_kw"] for r in checks),
        "max_analytic_error_kw": max(r["absolute_error_kw"] for r in analytical),
    }
    if report_path is not None:
        report_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    run()
