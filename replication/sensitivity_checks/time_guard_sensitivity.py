"""Focused timing, energy-conversion, and common-guard checks.

Run `python time_guard_sensitivity.py --freeze` before the first optimization,
then `python time_guard_sensitivity.py --run`. A rerun uses the same protocol.
Inputs are the preserved 24 case files; no fleet is replaced or silently dropped.

The exact-event model has binary availability and unequal interval lengths.
Power is constant within each interval; energy rows use that interval's duration.
Both endpoints of each common coefficient interval and every withdrawal subset
of cardinality q are enumerated. Since discharge is nonnegative, these also
protect every subset of size less than q. Cycle removal is applied afterwards.
This is an independent capacity calculation, not a use of an equal-slot cut
formula on fractional availability.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo

import numpy as np
import scipy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cplex_lp import linprog
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "capacity_checks" / "inputs" / "withdrawal_budget_cases"
PROTOCOL = HERE / "sensitivity_protocol.json"
TOL = 1e-7


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    payload = {
        "study": "Fixed-cohort timing, energy conversion, and no-discharge guard sensitivity",
        "frozen_utc_before_sensitivity_optimization": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(__file__),
        "case_sha256": {p.name: sha256(p) for p in sorted(INPUTS.glob("W*.json"))},
        "selection": "Same 24 fleets and all 144 physical session keys as the frozen primary cohort; no reselection",
        "factorial_design": {
            "timing": ["hourly_outward", "exact_events"],
            "energy": ["reported_as_battery_increment", "grid_consistent_increment"],
            "q": list(range(7)),
            "grid_consistent_increment": "Etarget = E0 + eta_c * reported_grid_kWh; preserve all other original assigned battery bounds and powers",
            "reported_as_battery_increment": "Keep original Etarget = E0 + reported_grid_kWh",
            "exact_events": "Clip source arrival/departure to local America/Los_Angeles horizon; partition by all clipped events and hourly boundaries; binary availability throughout each interval",
            "capacity_interpretation": "Instantaneous upper capacity for a piecewise-constant schedule; no energy-averaged capacity relaxation",
            "infeasible_policy": "Report every infeasible fleet/q and individual full-power charge deficit; do not adjust requests or replace sessions",
        },
        "fixed_parameters": "Original E0/Emin/Emax/Pc/Pd/eta_c/eta_d/L, common u endpoints [1,1.5], guard at every q",
        "guard_control": "For original hourly/battery-increment cases only, solve q=0 both with and without the no-discharge guard; removal is not used for q>=1",
        "guard_example": "Two slots; recipient connected first only with 1kWh demand; donor connected both with E0=target=1kWh, Emax=1+H, H in {0,0.25,0.5,1}; unit efficiencies, powers1kW, L=0,u=1; compare q=0 guarded/unguarded and q=1 guarded",
        "solver": {"package": "scipy.optimize.linprog", "method": "highs", "primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8},
        "verification": "Recover complementarity using (c,d)->(c-s,d-eta_c*eta_d*s). Audit power rows/bounds in kW separately from energy/target/guard rows in kWh.",
        "interpretation": "Sensitivity of a constructed retrospective cohort; no observed V2G claim and no equal-slot critical-cut certificate claim for unequal intervals",
    }
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text())
        if old["source_sha256"] != payload["source_sha256"] or old["case_sha256"] != payload["case_sha256"]:
            raise RuntimeError("Frozen protocol/code/input mismatch. Preserve this protocol and record an explicit amendment.")
        print("Frozen protocol already matches; preserved its timestamp.")
    else:
        PROTOCOL.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"protocol": str(PROTOCOL), "frozen_utc": payload["frozen_utc_before_sensitivity_optimization"]}))


def source_sessions(case):
    tz = ZoneInfo("America/Los_Angeles")
    origin = datetime.fromisoformat(case["metadata"]["horizon_date"]).replace(tzinfo=tz)
    end = origin + timedelta(days=1)
    seconds = (end.astimezone(timezone.utc) - origin.astimezone(timezone.utc)).total_seconds()
    if seconds != 86400:
        raise AssertionError("The fixed cohort must exclude DST horizons")
    rows = []
    for i, key in enumerate(case["metadata"]["selected_physical_keys"]):
        arrive, depart, energy = key.split("|")
        lo = max(0., (datetime.fromisoformat(arrive) - origin).total_seconds())
        hi = min(seconds, (datetime.fromisoformat(depart) - origin).total_seconds())
        if not 0 <= lo < hi <= seconds:
            raise AssertionError((case["name"], i, lo, hi))
        rows.append({"i": i, "key": key, "arrival_seconds": lo, "departure_seconds": hi, "grid_kwh": float(energy)})
        if abs(case["Etarget"][i] - case["E0"][i] - float(energy)) > 1e-9:
            raise AssertionError("Unexpected original energy construction")
    return rows


def build_variant(case, timing, energy):
    sessions = source_sessions(case)
    if timing == "hourly_outward":
        boundaries = np.arange(25, dtype=float) * 3600
        availability = np.asarray(case["availability"], dtype=float)
        # Verify the given data implements the stated outward convention.
        expected = np.zeros_like(availability)
        for row in sessions:
            lo = int(np.floor(row["arrival_seconds"] / 3600))
            hi = int(np.ceil(row["departure_seconds"] / 3600))
            expected[row["i"], lo:hi] = 1
        if not np.array_equal(availability, expected):
            raise AssertionError((case["name"], "Availability differs from stated rounding"))
    elif timing == "exact_events":
        boundaries = np.unique(np.r_[np.arange(25, dtype=float) * 3600,
                                     [r["arrival_seconds"] for r in sessions],
                                     [r["departure_seconds"] for r in sessions]])
        mid = (boundaries[:-1] + boundaries[1:]) / 2
        availability = np.array([[float(r["arrival_seconds"] <= t < r["departure_seconds"]) for t in mid] for r in sessions])
    else:
        raise ValueError(timing)
    duration = np.diff(boundaries) / 3600
    mid = (boundaries[:-1] + boundaries[1:]) / 2
    hour = np.floor(mid / 3600).astype(int)
    target = np.asarray(case["Etarget"], float)
    if energy == "grid_consistent_increment":
        target = np.asarray(case["E0"], float) + np.asarray(case["eta_c"], float) * np.array([r["grid_kwh"] for r in sessions])
    elif energy != "reported_as_battery_increment":
        raise ValueError(energy)
    return {"name": case["name"], "N": case["N"], "T": len(duration),
            "availability": availability, "duration": duration,
            "boundaries_seconds": boundaries, "L": np.asarray(case["L"], float)[hour],
            "Etarget": target,
            **{k: np.asarray(case[k], float) for k in ("E0", "Emin", "Emax", "Pc", "Pd", "eta_c", "eta_d")}}


def solve(model, q, guard=True, endpoints=(1., 1.5)):
    n, nt = model["availability"].shape
    nz = n * nt
    nv = 2 * nz + 1
    row_i, col_j, data, rhs, energy_rows, power_rows = [], [], [], [], [], []

    def add(indices, values, limit, unit):
        ix = len(rhs)
        row_i.extend([ix] * len(indices)); col_j.extend(indices); data.extend(values)
        rhs.append(float(limit))
        (energy_rows if unit == "energy" else power_rows).append(ix)

    duration = model["duration"]
    for i in range(n):
        for t in range(nt):
            cix = np.arange(i * nt, i * nt + t + 1)
            dix = cix + nz
            vals_c = duration[:t + 1] * model["eta_c"][i]
            vals_d = -duration[:t + 1] / model["eta_d"][i]
            ix = np.r_[cix, dix]
            vals = np.r_[vals_c, vals_d]
            add(ix, vals, model["Emax"][i] - model["E0"][i], "energy")
            add(ix, -vals, model["E0"][i] - model["Emin"][i], "energy")
            if guard:
                add(cix, vals_c, model["Emax"][i] - model["E0"][i], "energy")
        add(ix, -vals, model["E0"][i] - model["Etarget"][i], "energy")
    for w in combinations(range(n), q):
        retained = np.ones(n); retained[list(w)] = 0
        for t in range(nt):
            ix = np.r_[np.arange(n) * nt + t, nz + np.arange(n) * nt + t, nv - 1]
            for u in endpoints:
                add(ix, np.r_[np.repeat(u, n), -u * retained, -1], -model["L"][t], "power")
    matrix = coo_matrix((data, (row_i, col_j)), shape=(len(rhs), nv)).tocsr()
    rhs = np.asarray(rhs)
    upper = np.r_[(model["availability"] * model["Pc"][:, None]).ravel(),
                  (model["availability"] * model["Pd"][:, None]).ravel(), np.inf]
    bounds = [(0, float(x)) if np.isfinite(x) else (0, None) for x in upper]
    objective = np.zeros(nv); objective[-1] = 1
    ans = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=bounds)
    full_charge = model["eta_c"] * model["Pc"] * (model["availability"] @ duration)
    deficits = np.maximum(0, model["Etarget"] - model["E0"] - full_charge)
    result = {"status": "optimal" if ans.success else "infeasible" if ans.status == 2 else "solver_error",
              "cplex_status": ans.native_status, "M_kw": None,
              "individual_charge_deficit_kwh": float(np.max(deficits)),
              "energy_residual_kwh": None, "power_residual_kw": None, "overlap_kw": None,
              "minimum_subinterval_hours": float(np.min(duration)), "interval_count": nt}
    if not ans.success:
        if ans.status != 2:
            raise RuntimeError(ans.message)
        return result, None
    charge = ans.x[:nz].reshape(n, nt).copy()
    discharge = ans.x[nz:2*nz].reshape(n, nt).copy()
    k = (model["eta_c"] * model["eta_d"])[:, None]
    removed = np.minimum(charge, discharge / k)
    charge -= removed; discharge -= k * removed
    recovered = np.r_[charge.ravel(), discharge.ravel(), ans.fun]
    residual = matrix @ recovered - rhs
    energy_error = float(max(0., np.max(residual[energy_rows])))
    power_error = float(max(0., np.max(residual[power_rows]), np.max(-recovered), np.max(recovered-upper)))
    overlap = float(np.max(np.minimum(charge, discharge)))
    if energy_error > TOL or power_error > TOL or overlap > TOL:
        raise AssertionError((model["name"], q, energy_error, power_error, overlap))
    result.update(M_kw=float(ans.fun), energy_residual_kwh=energy_error, power_residual_kw=power_error, overlap_kw=overlap)
    return result, {"charge_kw": charge, "discharge_kw": discharge,
                    "energy_kwh": model["E0"][:, None] + np.cumsum(duration[None, :] * (model["eta_c"][:, None] * charge - discharge / model["eta_d"][:, None]), axis=1),
                    "no_discharge_energy_kwh": model["E0"][:, None] + np.cumsum(duration[None, :] * model["eta_c"][:, None] * charge, axis=1)}


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def headroom_example():
    rows, schedules = [], {}
    for headroom in (0., .25, .5, 1.):
        model = {"name": "fixed_charge_headroom_example", "N": 2, "T": 2,
                 "availability": np.array([[1., 0.], [1., 1.]]), "duration": np.ones(2),
                 "L": np.zeros(2), "E0": np.array([0., 1.]),
                 "Emin": np.zeros(2), "Emax": np.array([1., 1.+headroom]),
                 "Etarget": np.ones(2), "Pc": np.ones(2), "Pd": np.ones(2),
                 "eta_c": np.ones(2), "eta_d": np.ones(2)}
        for q, guard in ((0, True), (0, False), (1, True)):
            result, schedule = solve(model, q, guard=guard, endpoints=(1.,))
            expected = (max(.5, 1-headroom) if guard else .5) if q == 0 else 1.
            if result["status"] != "optimal" or abs(result["M_kw"] - expected) > TOL:
                raise AssertionError((headroom, q, guard, result, expected))
            row = {"headroom_kwh": headroom, "q": q, "guard": guard, **result, "analytic_M_kw": expected}
            rows.append(row)
            schedules[f"H{headroom}_q{q}_guard{int(guard)}"] = {k: v.tolist() for k, v in schedule.items()}
    write_csv("guard_headroom_example.csv", rows)
    (HERE / "guard_headroom_example_schedules.json").write_text(json.dumps(schedules, indent=2) + "\n")
    return rows


def run():
    start = time.time()
    if not PROTOCOL.exists():
        raise RuntimeError("Run --freeze before optimization")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["source_sha256"] != sha256(__file__):
        raise RuntimeError("Source changed after the protocol was frozen")
    cases = {p.stem: json.loads(p.read_text()) for p in sorted(INPUTS.glob("W*.json"))}
    if {p.name: sha256(p) for p in sorted(INPUTS.glob("W*.json"))} != protocol["case_sha256"]:
        raise RuntimeError("Source cases changed after the protocol was frozen")
    timing_rows, session_rows, guard_rows, schedules = [], [], [], {}
    for fleet, case in cases.items():
        for session in source_sessions(case):
            i = session["i"]
            original_h = sum(case["availability"][i])
            actual_h = (session["departure_seconds"] - session["arrival_seconds"]) / 3600
            session_rows.append({"fleet": fleet, "vehicle_1based": i+1, "physical_key": session["key"],
                "hourly_available_hours": original_h, "exact_available_hours": actual_h,
                "extra_hours": original_h-actual_h,
                "reported_grid_kwh": session["grid_kwh"],
                "original_required_grid_kwh": session["grid_kwh"]/case["eta_c"][i],
                "exact_max_grid_kwh": actual_h*case["Pc"][i],
                "original_battery_charge_deficit_kwh": max(0.,session["grid_kwh"]-actual_h*case["Pc"][i]*case["eta_c"][i]),
                "consistent_battery_charge_deficit_kwh": max(0.,case["eta_c"][i]*(session["grid_kwh"]-actual_h*case["Pc"][i]))})
        for timing in ("hourly_outward", "exact_events"):
            for energy in ("reported_as_battery_increment", "grid_consistent_increment"):
                model = build_variant(case, timing, energy)
                current = []
                for q in range(7):
                    result, schedule = solve(model, q)
                    row = {"fleet": fleet, "timing": timing, "energy_mapping": energy, "q": q, **result,
                           "saving_vs_same_variant_reservation_kw": None}
                    current.append(row)
                    if schedule is not None:
                        tag = f"{fleet}_{timing}_{energy}_q{q}"
                        schedules[tag+"__charge_kw"] = schedule["charge_kw"]
                        schedules[tag+"__discharge_kw"] = schedule["discharge_kw"]
                        schedules[tag+"__duration_hours"] = model["duration"]
                if len({r["status"] for r in current}) != 1:
                    raise AssertionError("Feasibility should not depend on q: unbounded M admits charge-only whenever individual energy is feasible")
                if current[-1]["status"] == "optimal":
                    mn = current[-1]["M_kw"]
                    for r in current:
                        r["saving_vs_same_variant_reservation_kw"] = mn-r["M_kw"]
                    if any(current[q]["M_kw"] > current[q+1]["M_kw"]+TOL for q in range(6)):
                        raise AssertionError("Nonmonotone frontier")
                timing_rows.extend(current)
        # This unguarded control is only a nominal q=0 comparison.
        model = build_variant(case, "hourly_outward", "reported_as_battery_increment")
        unguarded, schedule = solve(model, 0, guard=False)
        guarded = next(r for r in timing_rows if r["fleet"] == fleet and r["timing"] == "hourly_outward" and r["energy_mapping"] == "reported_as_battery_increment" and r["q"] == 0)
        if unguarded["status"] != "optimal":
            raise AssertionError("Original nominal cases should be feasible")
        excess = np.maximum(0., schedule["no_discharge_energy_kwh"]-model["Emax"][:, None])
        guard_rows.append({"fleet": fleet, "guarded_M0_kw": guarded["M_kw"], "unguarded_M0_kw": unguarded["M_kw"],
            "guard_capacity_cost_kw": guarded["M_kw"]-unguarded["M_kw"],
            "unguarded_optimum_no_discharge_upper_violation_kwh": float(np.max(excess)),
            "unguarded_energy_residual_kwh": unguarded["energy_residual_kwh"],
            "unguarded_power_residual_kw": unguarded["power_residual_kw"],
            "unguarded_overlap_kw": unguarded["overlap_kw"]})
        print(f"Completed {fleet}: four timing/energy variants and nominal guard control", flush=True)
    write_csv("timing_energy_frontiers.csv", timing_rows)
    write_csv("session_timing_audit.csv", session_rows)
    write_csv("guard_nominal_comparison.csv", guard_rows)
    np.savez_compressed(HERE / "recovered_schedules.npz", **schedules)
    example = headroom_example()

    # Direct verification against the separately reconstructed original frontier.
    with (HERE.parent / "capacity_checks" / "full_frontier_independent_lp.csv").open() as f:
        original = {(r["fleet"], int(r["q"])): float(r["M_kw"]) for r in csv.DictReader(f)}
    differences = [abs(r["M_kw"] - original[(r["fleet"], r["q"])]) for r in timing_rows if r["timing"] == "hourly_outward" and r["energy_mapping"] == "reported_as_battery_increment"]
    if max(differences) > TOL:
        raise AssertionError("New implementation does not reproduce the original frontier")
    summaries = []
    for timing in ("hourly_outward", "exact_events"):
        for energy in ("reported_as_battery_increment", "grid_consistent_increment"):
            sub = [r for r in timing_rows if r["timing"] == timing and r["energy_mapping"] == energy]
            feasible = sorted({r["fleet"] for r in sub if r["status"] == "optimal"})
            infeasible = sorted({r["fleet"] for r in sub if r["status"] == "infeasible"})
            counts = [sum(r["q"] == q and r["status"] == "optimal" and r["saving_vs_same_variant_reservation_kw"] > 1e-6 for r in sub) for q in range(7)]
            summaries.append({"timing": timing, "energy_mapping": energy,
                "feasible_fleet_count": len(feasible), "infeasible_fleets": infeasible,
                "positive_saving_counts_q0_to_q6": counts,
                "mean_saving_kw_on_feasible_fleets_q0_to_q6": [float(np.mean([r["saving_vs_same_variant_reservation_kw"] for r in sub if r["q"]==q and r["status"]=="optimal"])) for q in range(7)],
                "thresholds": [{"fleet": f, "first_zero_saving_q": min(r["q"] for r in sub if r["fleet"]==f and r["saving_vs_same_variant_reservation_kw"] <= 1e-6)} for f in feasible]})
    optimal = [r for r in timing_rows if r["status"] == "optimal"]
    summary = {
        "sensitivity_lp_calls": len(timing_rows), "optimal_sensitivity_lp_calls": len(optimal),
        "infeasible_sensitivity_lp_calls": len(timing_rows)-len(optimal),
        "nominal_unguarded_lp_calls": len(guard_rows), "headroom_example_lp_calls": len(example),
        "total_lp_calls": len(timing_rows)+len(guard_rows)+len(example),
        "source_session_count": len(session_rows),
        "mean_hourly_availability_hours": float(np.mean([r["hourly_available_hours"] for r in session_rows])),
        "mean_exact_availability_hours": float(np.mean([r["exact_available_hours"] for r in session_rows])),
        "mean_rounding_extra_hours": float(np.mean([r["extra_hours"] for r in session_rows])),
        "max_rounding_extra_hours": max(r["extra_hours"] for r in session_rows),
        "individually_infeasible_original_sessions": [r for r in session_rows if r["original_battery_charge_deficit_kwh"] > TOL],
        "individually_infeasible_consistent_sessions": [r for r in session_rows if r["consistent_battery_charge_deficit_kwh"] > TOL],
        "factorial_results": summaries,
        "guard_capacity_cost_positive_fleets": [r for r in guard_rows if r["guard_capacity_cost_kw"] > 1e-6],
        "max_guard_capacity_cost_kw": max(r["guard_capacity_cost_kw"] for r in guard_rows),
        "max_guard_difference_absolute_kw": max(abs(r["guard_capacity_cost_kw"]) for r in guard_rows),
        "max_energy_residual_kwh": max(r["energy_residual_kwh"] for r in optimal),
        "max_power_residual_kw": max(r["power_residual_kw"] for r in optimal),
        "max_overlap_kw": max(r["overlap_kw"] for r in optimal),
        "max_original_frontier_discrepancy_kw": max(differences),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "elapsed_seconds_not_a_benchmark": time.time()-start,
        "interpretation": "Sensitivity on fixed constructed inputs. Counts use only feasible fleets, with infeasible names explicitly retained. No out-of-sample validation."
    }
    (HERE / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "factorial_results"}, indent=2))
    for r in summaries:
        print(json.dumps({k:v for k,v in r.items() if k != "thresholds"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    if args.run:
        run()
    if not (args.freeze or args.run):
        parser.error("Choose --freeze or --run")
