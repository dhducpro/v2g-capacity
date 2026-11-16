"""Direct physical audit of saved schedules, independent of LP row assembly.

Uses sorted top-q discharge instead of the solver's subset enumeration and
checks connection times from source timestamps. No optimization is performed.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "capacity_checks" / "inputs" / "withdrawal_budget_cases"
rows = list(csv.DictReader((HERE / "timing_energy_frontiers.csv").open()))
archive = np.load(HERE / "recovered_schedules.npz")
records = []
for row in rows:
    if row["status"] != "optimal":
        continue
    case = json.loads((INPUTS / (row["fleet"] + ".json")).read_text())
    tag = f"{row['fleet']}_{row['timing']}_{row['energy_mapping']}_q{row['q']}"
    c, d, dt = (
        archive[tag + "__" + key] for key in ("charge_kw", "discharge_kw", "duration_hours")
    )
    ec, ed = np.array(case["eta_c"])[:, None], np.array(case["eta_d"])[:, None]
    e0 = np.array(case["E0"])[:, None]
    emin, emax = np.array(case["Emin"])[:, None], np.array(case["Emax"])[:, None]
    target = np.array(case["Etarget"])
    if row["energy_mapping"] == "grid_consistent_increment":
        target = np.array(case["E0"]) + ec[:, 0] * np.array(
            [float(k.split("|")[-1]) for k in case["metadata"]["selected_physical_keys"]]
        )
    origin = datetime.fromisoformat(case["metadata"]["horizon_date"]).replace(
        tzinfo=ZoneInfo("America/Los_Angeles")
    )
    source = [
        (
            max(0.0, (datetime.fromisoformat(k.split("|")[0]) - origin).total_seconds() / 3600),
            min(24.0, (datetime.fromisoformat(k.split("|")[1]) - origin).total_seconds() / 3600),
        )
        for k in case["metadata"]["selected_physical_keys"]
    ]
    if row["timing"] == "exact_events":
        events = np.unique(np.r_[np.arange(25), np.array(source).ravel()])
        assert np.max(abs(np.diff(events) - dt)) < 1e-12
        mid = (events[1:] + events[:-1]) / 2
        a = np.array([[float(s <= t < e) for t in mid] for s, e in source])
    else:
        assert np.array_equal(dt, np.ones(24))
        mid = np.arange(24) + 0.5
        a = np.array(case["availability"])
    energy = e0 + np.cumsum(dt[None, :] * (ec * c - d / ed), axis=1)
    guard = e0 + np.cumsum(dt[None, :] * ec * c, axis=1)
    energy_error = max(
        0.0,
        float(np.max(emin - energy)),
        float(np.max(energy - emax)),
        float(np.max(target - energy[:, -1])),
        float(np.max(guard - emax)),
    )
    q = int(row["q"])
    top_q = np.sort(d, axis=0)[-q:].sum(axis=0) if q else np.zeros(c.shape[1])
    net = c.sum(axis=0) - d.sum(axis=0) + top_q
    base = np.array(case["L"])[np.floor(mid).astype(int)]
    worst = np.maximum(base + net, base + 1.5 * net)
    power_error = max(
        0.0,
        float(np.max(-c)),
        float(np.max(-d)),
        float(np.max(c - a * np.array(case["Pc"])[:, None])),
        float(np.max(d - a * np.array(case["Pd"])[:, None])),
        float(np.max(worst - float(row["M_kw"]))),
    )
    overlap = float(np.max(np.minimum(c, d)))
    assert max(energy_error, power_error, overlap) < 1e-7, (tag, energy_error, power_error, overlap)
    records.append(
        {
            "case": tag,
            "energy_error_kwh": energy_error,
            "power_error_kw": power_error,
            "overlap_kw": overlap,
        }
    )
summary = {
    "saved_schedules_checked": len(records),
    "max_direct_energy_error_kwh": max(r["energy_error_kwh"] for r in records),
    "max_direct_power_error_kw": max(r["power_error_kw"] for r in records),
    "max_direct_overlap_kw": max(r["overlap_kw"] for r in records),
    "audit_method": "Original timestamps; energy trajectories; sorted top-q capacity rows; power bounds; complementarity. No LP row matrix reused.",
}
(HERE / "saved_schedule_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
