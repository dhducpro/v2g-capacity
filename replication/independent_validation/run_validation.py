"""Frozen validation of duration-weighted capacity certificates.

Freeze all inputs and implementations BEFORE the first study solve. Every
generated context and integer withdrawal budget is retained. Resume only when
the exact frozen protocol still matches. No outcome-based case replacement.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from weighted_capacity import (audit_schedule, compute_certificate,
                               reservation_cut, solve_capacity, solve_reservation)

HERE = Path(__file__).resolve().parent
REPLICATION = HERE.parent
PROTOCOL = HERE / "validation_protocol.json"
OUT = HERE / "results"
SAVING_TOL = 1e-6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bridge_inputs():
    """Revisit the known24 fleets with exact times and consistent energy."""
    source = REPLICATION / "sensitivity_checks/time_guard_sensitivity.py"
    spec = importlib.util.spec_from_file_location("frozen_timing_model", source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    destination = HERE / "inputs/bridge_cases"
    destination.mkdir(exist_ok=True)
    for p in sorted((REPLICATION / "capacity_checks/inputs/withdrawal_budget_cases").glob("W*.json")):
        original = json.loads(p.read_text())
        model = module.build_variant(original, "exact_events", "grid_consistent_increment")
        model = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()}
        model.update(name=p.stem, Ulo=[1.] * model["T"], Uhi=[1.5] * model["T"],
                     metadata={"cohort": "retrospective_exact_event_bridge",
                               "original_case_sha256": digest(p),
                               "new_independent_session_sample": False})
        dump(destination / p.name, model)


def input_paths():
    return [(cohort, p) for cohort, folder in [
        ("bridge", "bridge_cases"), ("temporal", "observed_cases"),
        ("mechanism", "synthetic_cases")]
        for p in sorted((HERE / "inputs" / folder).glob("*.json"))]


def source_paths():
    return [Path(__file__), HERE / "weighted_capacity.py",
            HERE / "build_independent_inputs.py", HERE / "contract_controls.py",
            REPLICATION / "sensitivity_checks/time_guard_sensitivity.py"]


def freeze():
    if PROTOCOL.exists():
        verify_protocol()
        print("Existing validation protocol matches; timestamp preserved.", flush=True)
        return
    data_protocol = json.loads((HERE / "data_protocol.json").read_text())
    for relative, expected in data_protocol["input_sha256"].items():
        if digest(HERE / "inputs" / relative) != expected:
            raise AssertionError((relative, "Frozen data mismatch"))
    bridge_inputs()
    inputs = input_paths()
    counts = {c: sum(group == c for group, _ in inputs) for c in ["bridge", "temporal", "mechanism"]}
    assert counts == {"bridge": 24, "temporal": 36, "mechanism": 96}, counts
    expected_calls = sum(json.loads(p.read_text())["N"] + 1 for _, p in inputs)
    payload = {
        "study": "Duration-consistent certificate validation and contract cost",
        "frozen_utc_before_study_optimization": datetime.now(timezone.utc).isoformat(),
        "data_protocol_sha256": digest(HERE / "data_protocol.json"),
        "data_frozen_before_optimization": data_protocol["frozen_utc_before_any_optimization"],
        "source_sha256": {str(p.relative_to(REPLICATION)): digest(p) for p in source_paths()},
        "input_sha256": {str(p.relative_to(HERE)): digest(p) for _, p in inputs},
        "cohorts": counts, "capacity_frontier_calls_planned": expected_calls,
        "reservation_calls_planned": len(inputs),
        "q_policy": "All integer q=0,...,N, including levels after certified collapse; no outcome-based stopping",
        "cut_policy": "Verified min-cut candidates selected only using charging reservation and input data; retain background-dominated, unresolved and infeasible contexts. No protected frontier outcomes used for cut selection.",
        "certificate_evaluation": "Best numerical upper bound over detected verified cuts at each q; first certified zero threshold determined from corrected bounds, not observed capacity savings. Not exhaustive critical-cut optimization.",
        "metrics": ["applicability of charging-driven critical cuts", "zero-certificate detection among actual zero observations", "fleet-level certified-minus-observed collapse gap", "occupancy comparator", "positive-bound-to-saving ratio", "missed zero cases and background-driven cases"],
        "replication_units": "36 temporal contexts on distinct dates; mechanism96 are eight paired variants of12 base configurations. q values are repeated decisions, not independent samples. No population confidence interval. Synthetic headroom/target/demand, surplus magnitude and varying L/U are joint operating regimes, not isolated causal effects. Temporal N groups use different prescribed dates, so no fixed-resource diversification effect is inferred from their comparison.",
        "saving_tolerance_kw": SAVING_TOL,
        "verification": "Direct recovered schedules; reservation equals q=N; nondecreasing frontier; no false zero within tolerance; bound validity; all168 bridge values compared with saved independent withdrawal-enumeration results.",
        "retention": "All predefined contexts retained, including adverse, infeasible and uninformative-certificate outcomes. Errors halt for correction with a logged implementation amendment; no input tuning.",
        "contract_control": "n4,D4,b.75,eta_c=eta_d=.9,L0,tau1; H{0,.25,.5,.75,1}; allq0..4; full fixed-charging LP versus independent scenario LP for restricted cancel-own-recharge policy; compare both with proved formulas.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    dump(PROTOCOL, payload)
    print(json.dumps({"protocol_frozen": payload["frozen_utc_before_study_optimization"],
                      "cohorts": counts, "planned_frontier_calls": expected_calls}), flush=True)


def verify_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    assert digest(HERE / "data_protocol.json") == protocol["data_protocol_sha256"]
    for relative, expected in protocol["source_sha256"].items():
        assert digest(REPLICATION / relative) == expected, (relative, "Source changed after freeze")
    for relative, expected in protocol["input_sha256"].items():
        assert digest(HERE / relative) == expected, (relative, "Input changed after freeze")
    return protocol


def original_bridge_values():
    with (REPLICATION / "sensitivity_checks/timing_energy_frontiers.csv").open() as stream:
        return {(r["fleet"], int(r["q"])): float(r["M_kw"]) for r in csv.DictReader(stream)
                if r["timing"] == "exact_events" and r["energy_mapping"] == "grid_consistent_increment"
                and r["status"] == "optimal"}


def run_one(cohort, path, protocol_hash, bridge):
    model = json.loads(path.read_text())
    name, n = model["name"], model["N"]
    destination = OUT / "cases" / (name + ".json")
    if destination.exists():
        previous = json.loads(destination.read_text())
        assert previous["validation_protocol_sha256"] == protocol_hash
        assert previous["input_sha256"] == digest(path)
        return previous
    start = time.time()
    reservation, reservation_schedule = solve_reservation(model)
    if reservation["status"] == "optimal":
        extraction = reservation_cut(model, reservation["M_kw"])
    else:
        extraction = {"status": "infeasible_reservation", "cuts": [], "selected_cut": None}
    # Build the complete input-based certificate before protected solves.
    certificates = []
    for q in range(n + 1):
        candidates = [compute_certificate(model, cut, q) for cut in extraction["cuts"]]
        best = min(candidates, key=lambda x: x["bound_kw"]) if candidates else None
        certificates.append({"q": q, "candidates": candidates, "best": best})
    rows, schedules = [], {}
    for q in range(n + 1):
        result, schedule = solve_capacity(model, q)
        cert = certificates[q]["best"]
        saving = reservation["M_kw"] - result["M_kw"] if result["status"] == reservation["status"] == "optimal" else None
        error = abs(result["M_kw"] - bridge[(name, q)]) if cohort == "bridge" and result["status"] == "optimal" else None
        row = {"cohort": cohort, "case": name, "N": n, "q": q,
               "status": result["status"], "M_kw": result["M_kw"],
               "M_reservation_kw": reservation["M_kw"], "saving_kw": saving,
               "certificate_status": extraction["status"],
               "bound_kw": cert["bound_kw"] if cert else None,
               "certifies_zero": bool(cert and cert["certifies_zero_saving"]),
               "energy_residual_kwh": result["energy_residual_kwh"],
               "power_residual_kw": result["power_residual_kw"],
               "overlap_kw": result["overlap_kw"], "bridge_error_kw": error,
               "capacity_solve_seconds": result["elapsed_seconds"]}
        if schedule is not None:
            for key, value in schedule.items(): schedules[f"q{q}_{key}"] = value
        if saving is not None:
            assert saving >= -SAVING_TOL, (name, q, "Reservation lower than protected value")
            if cert:
                assert saving <= cert["bound_kw"] + SAVING_TOL, (name, q, "Bound violation", saving, cert)
                assert not row["certifies_zero"] or saving <= SAVING_TOL, (name, q, "False zero", saving)
            if error is not None: assert error <= SAVING_TOL, (name, q, "Bridge mismatch", error)
        rows.append(row)
    if reservation["status"] == "optimal":
        assert all(r["status"] == "optimal" for r in rows), name
        assert abs(rows[-1]["M_kw"] - reservation["M_kw"]) <= SAVING_TOL
        assert all(b["M_kw"] >= a["M_kw"] - SAVING_TOL for a, b in zip(rows, rows[1:])), name
    else:
        assert all(r["status"] == "infeasible" for r in rows), name
    positive = [r for r in rows if r["saving_kw"] is not None and r["saving_kw"] > SAVING_TOL]
    zeros = [r for r in rows if r["saving_kw"] is not None and r["saving_kw"] <= SAVING_TOL]
    qzero = min((r["q"] for r in zeros), default=None)
    qcert = min((r["q"] for r in rows if r["certifies_zero"]), default=None)
    amax = int(np.asarray(model["availability"]).sum(axis=0).max())
    qocc = amax - 1 if reservation["status"] == "optimal" and reservation["M_kw"] > max(model["L"]) + SAVING_TOL else None
    summary = {"cohort": cohort, "case": name, "N": n,
               "intervals": model["T"], "reservation_status": reservation["status"],
               "M_reservation_kw": reservation["M_kw"], "cut_status": extraction["status"],
               "retained_cuts": len(extraction["cuts"]), "q_zero": qzero,
               "q_certificate": qcert, "q_occupancy": qocc,
               "threshold_gap": qcert - qzero if qcert is not None and qzero is not None else None,
               "occupancy_improvement": qocc - qcert if qocc is not None and qcert is not None else None,
               "positive_observations": len(positive), "zero_observations": len(zeros),
               "certified_zero_observations": sum(r["certifies_zero"] for r in rows),
               "nominal_saving_kw": rows[0]["saving_kw"],
               "one_withdrawal_saving_kw": rows[1]["saving_kw"],
               "elapsed_seconds": time.time() - start}
    record = {"validation_protocol_sha256": protocol_hash,
              "input_sha256": digest(path), "input_relative_path": str(path.relative_to(HERE)),
              "metadata": model.get("metadata", {}), "summary": summary,
              "reservation": reservation, "cut_extraction": extraction,
              "certificates": certificates, "frontier": rows}
    np.savez_compressed(OUT / "schedules" / (name + ".npz"), **schedules)
    dump(destination, record)
    print(json.dumps({k: summary[k] for k in ["cohort", "case", "N", "cut_status", "q_zero", "q_certificate", "q_occupancy", "elapsed_seconds"]}), flush=True)
    return record


def aggregate(records):
    frontiers = [r for record in records for r in record["frontier"]]
    cases = [record["summary"] for record in records]
    write_csv(OUT / "frontiers.csv", frontiers)
    write_csv(OUT / "case_summary.csv", cases)
    summary = {}
    for cohort in ["bridge", "temporal", "mechanism"]:
        cc = [r for r in cases if r["cohort"] == cohort]
        rr = [r for r in frontiers if r["cohort"] == cohort]
        eligible = [r for r in cc if r["q_certificate"] is not None]
        zero = [r for r in rr if r["saving_kw"] is not None and r["saving_kw"] <= SAVING_TOL]
        ratio = [r["bound_kw"] / r["saving_kw"] for r in rr if r["bound_kw"] is not None and r["saving_kw"] > SAVING_TOL]
        gaps = [r["threshold_gap"] for r in eligible]
        summary[cohort] = {"contexts": len(cc), "frontier_calls": len(rr),
            "optimal_contexts": sum(r["reservation_status"] == "optimal" for r in cc),
            "certificate_applicable_contexts": len(eligible),
            "background_dominated_contexts": sum(r["cut_status"] == "background_dominated_no_certificate" for r in cc),
            "unresolved_critical_cut_contexts": sum(r["cut_status"] == "unresolved_critical_cut" for r in cc),
            "exact_collapse_matches": sum(r["threshold_gap"] == 0 for r in eligible),
            "threshold_gap_median": float(np.median(gaps)) if gaps else None,
            "threshold_gap_max": max(gaps) if gaps else None,
            "strictly_improves_occupancy": sum(r["occupancy_improvement"] > 0 for r in eligible if r["occupancy_improvement"] is not None),
            "equals_occupancy": sum(r["occupancy_improvement"] == 0 for r in eligible if r["occupancy_improvement"] is not None),
            "worse_than_occupancy": sum(r["occupancy_improvement"] < 0 for r in eligible if r["occupancy_improvement"] is not None),
            "true_zero_fleet_q_observations": len(zero),
            "certified_zero_fleet_q_observations": sum(r["certifies_zero"] for r in zero),
            "zero_detection_fraction": sum(r["certifies_zero"] for r in zero) / len(zero) if zero else None,
            "zero_detection_excluding_trivial_qN": (sum(r["certifies_zero"] for r in zero if r["q"] < r["N"]) / sum(r["q"] < r["N"] for r in zero)) if any(r["q"] < r["N"] for r in zero) else None,
            "positive_bound_ratio_count": len(ratio),
            "positive_bound_ratio_min_median_max": [float(np.min(ratio)), float(np.median(ratio)), float(np.max(ratio))] if ratio else None,
            "positive_at_q0": sum(r["nominal_saving_kw"] is not None and r["nominal_saving_kw"] > SAVING_TOL for r in cc),
            "positive_at_q1": sum(r["one_withdrawal_saving_kw"] is not None and r["one_withdrawal_saving_kw"] > SAVING_TOL for r in cc)}
    optimal = [r for r in frontiers if r["status"] == "optimal"]
    audit = {"contexts": len(cases), "frontier_calls": len(frontiers),
             "optimal_calls": len(optimal), "infeasible_calls": sum(r["status"] == "infeasible" for r in frontiers),
             "false_zero_certificates": sum(r["certifies_zero"] and r["saving_kw"] > SAVING_TOL for r in optimal),
             "maximum_bound_violation_kw": max([0.] + [r["saving_kw"] - r["bound_kw"] for r in optimal if r["bound_kw"] is not None]),
             "maximum_bridge_error_kw": max([0.] + [r["bridge_error_kw"] for r in optimal if r["bridge_error_kw"] is not None]),
             **{f"maximum_{key}": max([0.] + [r[key] for r in optimal]) for key in ["energy_residual_kwh", "power_residual_kw", "overlap_kw"]}}
    dump(OUT / "validation_summary.json", {"cohorts": summary, "audit": audit,
        "interpretation": "Descriptive frozen-design validation; repeated q and paired scenarios are not independent replications; all adverse outcomes retained."})
    print(json.dumps({"final_audit": audit, "cohorts": summary}, indent=2), flush=True)


def run():
    protocol = verify_protocol()
    protocol_hash = digest(PROTOCOL)
    for folder in [OUT / "cases", OUT / "schedules"]: folder.mkdir(parents=True, exist_ok=True)
    bridge = original_bridge_values()
    records = [run_one(c, p, protocol_hash, bridge) for c, p in input_paths()]
    aggregate(records)
    assert sum(len(r["frontier"]) for r in records) == protocol["capacity_frontier_calls_planned"]


def audit_saved():
    verify_protocol()
    all_records = []
    maxima = {"energy_residual_kwh": 0., "power_residual_kw": 0., "overlap_kw": 0.}
    checked = 0
    for cohort, p in input_paths():
        model = json.loads(p.read_text())
        record = json.loads((OUT / "cases" / (model["name"] + ".json")).read_text())
        assert record["validation_protocol_sha256"] == digest(PROTOCOL)
        assert record["input_sha256"] == digest(p)
        with np.load(OUT / "schedules" / (model["name"] + ".npz")) as saved:
            for row in record["frontier"]:
                if row["status"] != "optimal": continue
                q = row["q"]
                audit = audit_schedule(model, q, saved[f"q{q}_charge_kw"], saved[f"q{q}_discharge_kw"], row["M_kw"])
                for key in maxima: maxima[key] = max(maxima[key], audit[key])
                assert max(audit[k] for k in maxima) <= SAVING_TOL
                checked += 1
        all_records.append(record)
    result = {"status": "PASS", "saved_schedules_checked": checked,
              "independent_of_lp_matrix": True, "maxima": maxima}
    dump(OUT / "saved_schedule_audit.json", result)
    aggregate(all_records)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit-saved", action="store_true")
    args = parser.parse_args()
    if args.freeze: freeze()
    elif args.run: run()
    else: audit_saved()
