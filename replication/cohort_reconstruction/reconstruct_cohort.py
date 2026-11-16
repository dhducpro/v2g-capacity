#!/usr/bin/env python3
"""Rebuild the frozen 24-fleet cohort from source records, without optimization.

Run from any directory with Python 3.10+ (standard library only):
    python reconstruct_cohort.py

This focused implementation follows the input-selection part of the historical
withdrawal_budget_experiments.py whose SHA256 is recorded in the frozen protocol.
It includes no economic, risk-evaluation, or solver code. A successful run checks
the input hashes, 28 date-selection decisions, all 144 selected records, and all
24 JSON case hashes against the unchanged frozen cohort. Rebuilt files are put
in this directory's `reconstructed` subdirectory; frozen inputs are never edited.

The original export's naive timestamps are interpreted in America/Los_Angeles.
The dates reached by this selection are January 1--10, February 1--9, and March
1--9, 2019; none crosses a DST transition. UTC arithmetic is used for durations.
Tariff fields are reproduced only to recover historical JSON byte identity; they
have no role in the capacity objective or in session/date selection.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
FROZEN = ROOT.parent / "capacity_checks" / "inputs"
LOCAL = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc
SEED = 2026111306
N = 6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=LOCAL) if parsed.tzinfo is None else parsed.astimezone(LOCAL)


def key(a: datetime, b: datetime, energy: float) -> str:
    return "|".join(
        (a.astimezone(UTC).isoformat(), b.astimezone(UTC).isoformat(), format(energy, ".12g"))
    )


def hours_between(a: datetime, b: datetime) -> float:
    return (b.astimezone(UTC) - a.astimezone(UTC)).total_seconds() / 3600.0


def source_pool() -> tuple[list[dict], dict]:
    source = read_csv(INPUTS / "caltech.csv")
    begin, end = datetime(2019, 1, 1, tzinfo=LOCAL), datetime(2019, 4, 1, tzinfo=LOCAL)
    records = []
    for index, row in enumerate(source):
        try:
            a, b = parse_time(row["Start"]), parse_time(row["End"])
            energy = float(row["Energy"])
        except (ValueError, TypeError):
            continue
        if not math.isfinite(energy) or energy < 2 or not begin <= a < end or b <= a:
            continue
        records.append(
            {
                "source_export_row_identifier": str(index),
                "connection_local": a,
                "disconnect_local": b,
                "energy": energy,
                "physical_key": key(a, b, energy),
            }
        )
    # Stable sorting and first occurrence match the historical pandas code.
    unique = {}
    for record in sorted(records, key=lambda x: x["physical_key"]):
        unique.setdefault(record["physical_key"], record)
    return list(unique.values()), {
        "original_source_rows": len(source),
        "quarter_records_before_deduplication": len(records),
        "quarter_unique_eligible_energy_records": len(unique),
    }


def make_case(
    chosen: list[dict], fleet: str, month: int, date: str, prices: list[float]
) -> tuple[dict, list[dict]]:
    availability = [[0.0] * 24 for _ in range(N)]
    e0, target, emax, selected = [], [], [], []
    for i, row in enumerate(chosen):
        energy = row["energy"]
        initial_size = max(55.0, min(100.0, energy / 0.45 + 20.0))
        initial = 0.4 * initial_size
        departure = initial + energy
        maximum = max(initial_size, departure / 0.92)
        e0.append(initial)
        target.append(departure)
        emax.append(maximum)
        for t in range(row["arrival"], row["departure"]):
            availability[i][t] = 1.0
        selected.append(
            {
                "fleet": fleet,
                "month": month,
                "horizon_date": date,
                "vehicle_index": i,
                "source_export_row_identifier": row["source_export_row_identifier"],
                "physical_key": row["physical_key"],
                "selection_hash": row["selection_hash"],
                "connection_local": str(row["connection_local"]),
                "disconnect_local": str(row["disconnect_local"]),
                "arrival_slot_zero_based": row["arrival"],
                "departure_slot_exclusive": row["departure"],
                "available_slots": row["departure"] - row["arrival"],
                "reported_delivered_kwh_as_required_battery_increment": energy,
                "E0_kwh": initial,
                "Etarget_kwh": departure,
                "Emin_kwh": 0.1 * maximum,
                "Emax_kwh": maximum,
                "Pc_kw": 7.2,
                "Pd_kw": 4.0,
            }
        )
    metadata = {
        "fleet": fleet,
        "month": month,
        "horizon_date": date,
        "selected_physical_keys": [row["physical_key"] for row in chosen],
        "source_export_row_identifiers": [row["source_export_row_identifier"] for row in chosen],
        "selection_seed": SEED,
        "constructed_hybrid_case": True,
        "common_full_withdrawal_battery_headroom_guard": True,
        "price_date_UTC": "2019-04-22",
    }
    # Key order and JSON formatting reproduce revision_model.Case serialization.
    case = {
        "N": N,
        "T": 24,
        "L": [10.0] * 24,
        "availability": availability,
        "price_buy": [[p + 0.002 / 2 for p in prices] for _ in range(N)],
        "price_sell": [[p - 0.002 / 2 for p in prices] for _ in range(N)],
        "u_nominal": [[1.5] * 24 for _ in range(N)],
        "Pc": [7.2] * N,
        "Pd": [4.0] * N,
        "E0": e0,
        "Emin": [0.1 * e for e in emax],
        "Emax": emax,
        "Etarget": target,
        "eta_c": [0.95] * N,
        "eta_d": [0.9] * N,
        "rho": [0.001] * N,
        "M": 100.0,
        "alpha": 0.0002,
        "dt": 1.0,
        "name": "withdrawal_budget_" + fleet,
        "metadata": metadata,
        "affine_capacity_rows": [],
    }
    return case, selected


def reconstruct() -> tuple[dict, list[dict], list[dict], list[dict], dict]:
    work, counts = source_pool()
    excluded = read_csv(INPUTS / "original_excluded_sessions.csv")
    used = {
        key(
            parse_time(row["connection_local"]),
            parse_time(row["disconnect_local"]),
            float(row["reported_delivered_kwh_as_required_battery_increment"]),
        )
        for row in excluded
    }
    old_dates = {row["horizon_date"] for row in excluded}
    counts.update(
        {"original_excluded_keys": len(used), "original_excluded_dates": sorted(old_dates)}
    )
    prices = [
        float(row["raw_price_eur_per_mwh"]) / 1000.0
        for row in read_csv(INPUTS / "frozen_tariff_profile.csv")
    ]
    if len(prices) != 24 or not all(math.isfinite(x) for x in prices):
        raise ValueError("The frozen tariff requires 24 finite entries")
    cases, records, audit, candidate_audit = {}, [], [], []
    for month in (1, 2, 3):
        selected_count = 0
        for day in range(1, calendar.monthrange(2019, month)[1] + 1):
            start = datetime(2019, month, day, tzinfo=LOCAL)
            end = start + timedelta(days=1)
            date = start.strftime("%Y-%m-%d")
            item = {"month": month, "date": date, "selected": False}
            if date in old_dates:
                audit.append({**item, "reason": "original_integrated_horizon"})
                continue
            if hours_between(start, end) != 24:
                audit.append({**item, "reason": "non_24_hour_local_horizon"})
                continue
            overlapping = [
                r for r in work if r["connection_local"] < end and r["disconnect_local"] > start
            ]
            item["overlapping_physical_sessions"] = len(overlapping)
            unused = [r for r in overlapping if r["physical_key"] not in used]
            item["after_prior_session_exclusion"] = len(unused)
            candidates = []
            for row in overlapping:
                a, b = max(row["connection_local"], start), min(row["disconnect_local"], end)
                arrival = max(0, min(23, math.floor(hours_between(start, a))))
                departure = max(1, min(24, math.ceil(hours_between(start, b))))
                previously_used = row["physical_key"] in used
                energy_feasible = departure > arrival and row["energy"] <= 0.95 * 7.2 * (
                    departure - arrival
                )
                digest = hashlib.sha256(f"{SEED}|{date}|{row['physical_key']}".encode()).hexdigest()
                record = {
                    **row,
                    "arrival": arrival,
                    "departure": departure,
                    "selection_hash": digest,
                }
                if not previously_used and energy_feasible:
                    candidates.append(record)
                candidate_audit.append(
                    {
                        "horizon_date": date,
                        "physical_key": row["physical_key"],
                        "source_export_row_identifier": row["source_export_row_identifier"],
                        "previously_used": previously_used,
                        "hourly_energy_feasible": energy_feasible,
                        "arrival_slot_zero_based": arrival,
                        "departure_slot_exclusive": departure,
                        "reported_delivered_kwh": row["energy"],
                        "selection_hash": digest,
                        "selected": False,
                        "eligible_rank": "",
                    }
                )
            item["individually_energy_feasible"] = len(candidates)
            ranked = sorted(candidates, key=lambda r: (r["selection_hash"], r["physical_key"]))
            ranks = {r["physical_key"]: i + 1 for i, r in enumerate(ranked)}
            enough = len(candidates) >= N
            for record in candidate_audit:
                if record["horizon_date"] == date and record["physical_key"] in ranks:
                    record["eligible_rank"] = ranks[record["physical_key"]]
                    record["selected"] = enough and ranks[record["physical_key"]] <= N
            if not enough:
                audit.append({**item, "reason": "fewer_than_six_eligible_unused_sessions"})
                continue
            chosen = sorted(
                ranked[:N],
                key=lambda r: (r["connection_local"], r["disconnect_local"], r["physical_key"]),
            )
            selected_count += 1
            fleet = f"W{month}{selected_count:02d}"
            case, selected = make_case(chosen, fleet, month, date, prices)
            cases[fleet] = case
            records.extend(selected)
            used.update(r["physical_key"] for r in chosen)
            audit.append(
                {**item, "selected": True, "fleet": fleet, "reason": "first_eight_eligible_dates"}
            )
            if selected_count == 8:
                break
        if selected_count != 8:
            raise ValueError(
                f"Month {month}: expected eight eligible dates, found {selected_count}"
            )
    return cases, records, audit, candidate_audit, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reconstructed")
    args = parser.parse_args()
    protocol = json.loads((FROZEN / "withdrawal_budget_protocol.json").read_text())
    pin_map = {
        "caltech.csv": "acn_source_sha256",
        "original_excluded_sessions.csv": "old_selection_sha256",
        "frozen_selected_sessions.csv": "selected_records_sha256",
        "frozen_tariff_profile.csv": "tariff_source_sha256",
    }
    for filename, field in pin_map.items():
        if sha256(INPUTS / filename) != protocol[field]:
            raise ValueError(f"Hash mismatch for original input: {filename}")
    cases, records, audit, candidate_audit, counts = reconstruct()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_dir = output / "withdrawal_budget_cases"
    case_dir.mkdir(exist_ok=True)
    hashes, exact_case_content = {}, {}
    for fleet, case in cases.items():
        target = case_dir / f"{fleet}.json"
        target.write_text(json.dumps(case, indent=2), encoding="utf-8", newline="\n")
        hashes[fleet] = sha256(target)
        exact_case_content[fleet] = case == json.loads(
            (FROZEN / "withdrawal_budget_cases" / f"{fleet}.json").read_text()
        )
        if hashes[fleet] != protocol["case_sha256"][fleet]:
            raise ValueError(
                f"Reconstructed case hash differs: {fleet}; content match {exact_case_content[fleet]}"
            )
    selected_path = output / "withdrawal_budget_selected_sessions.csv"
    write_csv(selected_path, records)
    frozen_rows = read_csv(INPUTS / "frozen_selected_sessions.csv")
    rebuilt_rows = read_csv(selected_path)
    if rebuilt_rows != frozen_rows:
        differences = [
            (i, k, a.get(k), b.get(k))
            for i, (a, b) in enumerate(zip(rebuilt_rows, frozen_rows))
            for k in set(a) | set(b)
            if a.get(k) != b.get(k)
        ]
        raise ValueError(f"Reconstructed selected-session records differ: {differences[:5]}")
    frozen_audit = json.loads((INPUTS / "frozen_date_selection_audit.json").read_text())
    if audit != frozen_audit:
        raise ValueError("Reconstructed first-eligible date decisions differ")
    write_csv(output / "candidate_selection_audit.csv", candidate_audit)
    (output / "date_selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    summary = {
        "status": "PASS",
        "optimizer_called": False,
        "input_hashes_match_frozen_protocol": True,
        **counts,
        "date_decisions_reproduced": len(audit),
        "selected_fleets": len(cases),
        "selected_dates": [case["metadata"]["horizon_date"] for case in cases.values()],
        "selected_sessions": len(records),
        "unique_selected_physical_keys": len({r["physical_key"] for r in records}),
        "scanned_candidate_date_pairs": len(candidate_audit),
        "selected_records_all_fields_match": rebuilt_rows == frozen_rows,
        "selected_records_byte_identical": sha256(selected_path)
        == protocol["selected_records_sha256"],
        "date_audit_identical": audit == frozen_audit,
        "all_case_json_byte_hashes_match": all(
            hashes[k] == protocol["case_sha256"][k] for k in hashes
        ),
        "case_sha256": hashes,
        "scope": "Rebuilds the frozen hourly constructed cohort. It does not validate observed battery states, change the cohort, infer V2G behavior, or acquire new upstream data.",
    }
    (output / "RECONSTRUCTION_CHECK.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "case_sha256"}, indent=2))


if __name__ == "__main__":
    main()
