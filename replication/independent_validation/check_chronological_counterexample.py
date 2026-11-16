"""Analytical counterexample with a unique critical cut; not a cohort sample.

Added after interpretation review to distinguish a per-cut converse from a
converse after exhaustive critical-cut selection. Its exact constants and
expected values are recorded before its four optimization calls.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from weighted_capacity import solve_capacity, solve_reservation, reservation_cut, compute_certificate

HERE = Path(__file__).resolve().parent


def main():
    model = {"name": "unique_critical_cut_chronology_counterexample", "N": 3, "T": 2,
             "availability": [[1., 0.], [1., 1.], [1., 1.]], "duration": [1., 1.],
             "L": [0., 0.], "Ulo": [1., 1.], "Uhi": [1., 1.],
             "E0": [0., 0., 0.], "Emin": [0., 0., 0.],
             "Etarget": [1.8, 0., 0.], "Emax": [1.8, 3., 3.],
             "Pc": [2., 2., 2.], "Pd": [0., 1., 1.],
             "eta_c": [.9, .9, .9], "eta_d": [.9, .9, .9]}
    definition = {"role": "Analytically designed limitation, not an independent validation sample",
                  "model": model, "q": [0, 1, 2, 3], "expected_M_kw": [2., 2., 2., 2.],
                  "nonempty_subset_ratios_kw": {"1": 2., "2": 0., "1,2": 1.},
                  "unique_critical_slots_1based": [1], "expected_R_kwh": [2., 1., 0., 0.],
                  "expected_observed_zero_threshold": 0, "expected_best_critical_cut_threshold": 2}
    protocol_path = HERE / "chronological_counterexample_protocol.json"
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        assert old["source_sha256"] == source_hash and old["definition"] == definition
    else:
        protocol_path.write_text(json.dumps({"frozen_before_diagnostic_solves_utc": datetime.now(timezone.utc).isoformat(),
                                           "source_sha256": source_hash, "definition": definition}, indent=2) + "\n")
    a = np.array(model["availability"])
    h = (np.array(model["Etarget"]) - model["E0"]) / model["eta_c"]
    all_cuts = []
    for slots in ([0], [1], [0, 1]):
        outside = [t for t in range(2) if t not in slots]
        out = (a[:, outside] * np.array(model["Pc"])[:, None]).sum(axis=1)
        phi = float(np.maximum(h - out, 0.).sum())
        all_cuts.append({"slots_1based": [t + 1 for t in slots], "ratio_kw": phi / len(slots)})
    assert [r["ratio_kw"] for r in all_cuts] == [2., 0., 1.]
    reserve, _ = solve_reservation(model)
    cuts = reservation_cut(model, reserve["M_kw"])
    assert cuts["status"] == "verified_critical_cut"
    assert len(cuts["cuts"]) == 1 and cuts["selected_cut"]["slots"] == [0]
    rows = []
    for q in range(4):
        result, schedule = solve_capacity(model, q)
        certificate = compute_certificate(model, cuts["selected_cut"], q)
        assert abs(result["M_kw"] - 2.) <= 1e-8
        assert abs(certificate["R_kwh"] - definition["expected_R_kwh"][q]) <= 1e-8
        rows.append({"q": q, "capacity": result, "certificate": certificate,
                     "schedule": {key: value.tolist() for key, value in schedule.items()}})
    output = {"status": "PASS", "scope": definition["role"], "all_nonempty_cuts": all_cuts,
              "reservation": reserve, "unique_critical_cut": cuts["selected_cut"],
              "observed_zero_threshold": 0, "best_critical_cut_threshold": 2,
              "capacity_calls": 4, "reservation_calls": 1,
              "frontier": rows}
    (HERE / "results/chronological_counterexample.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "unique_critical_cut": [1],
                      "M_kw": [r["capacity"]["M_kw"] for r in rows],
                      "R_kwh": [r["certificate"]["R_kwh"] for r in rows],
                      "observed_zero_threshold": 0, "best_critical_cut_threshold": 2}))


if __name__ == "__main__":
    main()
