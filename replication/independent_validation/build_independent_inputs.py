#!/usr/bin/env python3
"""Build the outcome-independent round-10 validation inputs.

No optimization or capacity-result file is imported by this program. Run
``python build_independent_inputs.py --freeze`` before solving any of the
generated fleets. Subsequent ``--check`` runs reconstruct all inputs in memory
and compare their hashes with the retained protocol and files.

The observed-session cohort uses the supplied processed ACN export, not a new
site or newly downloaded raw records. Only full-session timing and delivered
grid energy are observed. Battery attributes and V2G powers are assumptions.
The synthetic factorial has 96 scenarios on 12 paired base configurations;
these are explicitly not 96 independent replications.
"""
from __future__ import annotations

import argparse
import calendar
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
from itertools import product
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "cohort_reconstruction" / "inputs"
OUTPUT = HERE / "inputs"
PROTOCOL = HERE / "data_protocol.json"
LOCAL = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc
SEED = 20261114
SIZES = (6, 12, 24)
MONTHS = tuple(range(4, 10))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest_token(token: str) -> str:
    return hashlib.sha256(f"{SEED}|{token}".encode()).hexdigest()


def uniform(token: str) -> float:
    # Every draw is a stable keyed hash; no dependence on iteration order or
    # NumPy/Python PRNG versions is introduced.
    return int(digest_token(token)[:16], 16) / 2**64


def choice(token: str, values):
    return values[int(digest_token(token), 16) % len(values)]


def parse_time(value: str) -> datetime:
    time = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return time.replace(tzinfo=LOCAL) if time.tzinfo is None else time.astimezone(LOCAL)


def hours_between(start: datetime, end: datetime) -> float:
    return (end.astimezone(UTC) - start.astimezone(UTC)).total_seconds() / 3600


def physical_key(start: datetime, end: datetime, energy: float) -> str:
    return "|".join((start.astimezone(UTC).isoformat(), end.astimezone(UTC).isoformat(), format(energy, ".12g")))


def exclusions() -> set[str]:
    excluded = {r["physical_key"] for r in rows(SOURCE / "frozen_selected_sessions.csv")}
    for row in rows(SOURCE / "original_excluded_sessions.csv"):
        excluded.add(physical_key(parse_time(row["connection_local"]), parse_time(row["disconnect_local"]),
                                  float(row["reported_delivered_kwh_as_required_battery_increment"])))
    assert len(excluded) == 162
    return excluded


def profiles(boundaries: list[float], kind: str) -> tuple[list[float], list[float], list[float]]:
    hours = [int((lo + hi) / 7200) for lo, hi in zip(boundaries[:-1], boundaries[1:])]
    if kind == "constant":
        return [10.0] * len(hours), [1.0] * len(hours), [1.0] * len(hours)
    if kind != "varying":
        raise ValueError(kind)
    load = [6 + 4 * math.exp(-((hour - 16) / 4)**2) for hour in hours]
    lower = [.9] * len(hours)
    upper = [1.1 + .3 * math.sin(math.pi * (hour - 7) / 24)**2 for hour in hours]
    return load, lower, upper


def make_model(name: str, vehicles: list[dict], profile: str, metadata: dict) -> dict:
    boundaries = sorted({float(hour * 3600) for hour in range(25)} |
                        {float(v[key]) for v in vehicles for key in ("arrival_seconds", "departure_seconds")})
    duration = [(hi - lo) / 3600 for lo, hi in zip(boundaries[:-1], boundaries[1:])]
    middle = [(lo + hi) / 2 for lo, hi in zip(boundaries[:-1], boundaries[1:])]
    available = [[int(v["arrival_seconds"] <= t < v["departure_seconds"]) for t in middle] for v in vehicles]
    load, lower, upper = profiles(boundaries, profile)
    model = {"name": name, "N": len(vehicles), "T": len(duration),
             "availability": available, "duration": duration, "boundaries_seconds": boundaries,
             "L": load, "Ulo": lower, "Uhi": upper}
    for key in ("E0", "Emin", "Emax", "Etarget", "Pc", "Pd", "eta_c", "eta_d"):
        model[key] = [v[key] for v in vehicles]
    model["metadata"] = metadata
    for i, vehicle in enumerate(vehicles):
        assert 0 <= vehicle["arrival_seconds"] < vehicle["departure_seconds"] <= 86400
        assert vehicle["Emin"] - 1e-10 <= vehicle["E0"] <= vehicle["Emax"] + 1e-10
        assert vehicle["Emin"] - 1e-10 <= vehicle["Etarget"] <= vehicle["Emax"] + 1e-10
        available_hours = sum(dt * a for dt, a in zip(duration, available[i]))
        assert abs(available_hours - (vehicle["departure_seconds"] - vehicle["arrival_seconds"]) / 3600) <= 1e-10
        # This is the declared input eligibility/construction check, not an
        # optimization-based exclusion or a claim of station feasibility.
        deficit = vehicle["Etarget"] - vehicle["E0"] - vehicle["eta_c"] * vehicle["Pc"] * available_hours
        assert deficit <= 1e-9, (name, i, deficit)
    assert all(dt > 0 for dt in duration) and abs(sum(duration) - 24) < 1e-10
    return model


def observed_pool() -> tuple[dict[str, list[dict]], list[dict], dict]:
    excluded = exclusions()
    source_rows = rows(SOURCE / "caltech.csv")
    seen = set()
    pool: dict[str, list[dict]] = {}
    audit = []
    begin, finish = datetime(2019, 4, 1, tzinfo=LOCAL), datetime(2019, 10, 1, tzinfo=LOCAL)
    for index, row in enumerate(source_rows):
        try:
            start, end, energy = parse_time(row["Start"]), parse_time(row["End"]), float(row["Energy"])
        except (ValueError, TypeError):
            continue
        if not begin <= start < finish:
            continue
        key = physical_key(start, end, energy)
        date = str(start.date())
        origin = datetime.combine(start.date(), datetime.min.time(), tzinfo=LOCAL)
        day_end = origin + timedelta(days=1)
        pc = choice("observed|Pc|" + key, (3.6, 7.2, 11.0))
        record = {"source_export_row_identifier": index, "horizon_date": date,
                  "physical_key": key, "connection_local": start.isoformat(),
                  "disconnect_local": end.isoformat(), "reported_grid_kwh": energy,
                  "assigned_Pc_kw": pc, "eligibility": "eligible"}
        if not math.isfinite(energy) or energy < 2:
            record["eligibility"] = "nonfinite_or_less_than_2_kwh"
        elif end <= start:
            record["eligibility"] = "nonpositive_session_duration"
        elif end > day_end:
            record["eligibility"] = "session_not_fully_contained_in_day"
        elif hours_between(origin, day_end) != 24:
            record["eligibility"] = "non_24_hour_local_horizon"
        elif key in seen:
            record["eligibility"] = "duplicate_physical_key"
        elif key in excluded:
            record["eligibility"] = "previously_used_physical_key"
        elif energy > pc * hours_between(start, end) + 1e-9:
            record["eligibility"] = "observed_energy_exceeds_assigned_charger_deliverability"
        seen.add(key)
        audit.append(record)
        if record["eligibility"] != "eligible":
            continue
        vehicle = {"physical_key": key, "source_export_row_identifier": index,
                   "connection_local": start.isoformat(), "disconnect_local": end.isoformat(),
                   "arrival_seconds": hours_between(origin, start) * 3600,
                   "departure_seconds": hours_between(origin, end) * 3600,
                   "reported_grid_kwh": energy, "Pc": pc,
                   "Pd": choice("observed|Pd|" + key, (2.0, 4.0, 7.2)),
                   "eta_c": choice("observed|eta_c|" + key, (.90, .93, .96)),
                   "eta_d": choice("observed|eta_d|" + key, (.88, .92, .96)),
                   "E0": choice("observed|E0|" + key, (15.0, 25.0, 35.0)),
                   "Emin": 0.0,
                   "headroom_kwh": choice("observed|headroom|" + key, (0.0, .5, 5.0, 15.0))}
        vehicle["Etarget"] = vehicle["E0"] + vehicle["eta_c"] * energy
        vehicle["Emax"] = vehicle["Etarget"] + vehicle["headroom_kwh"]
        pool.setdefault(date, []).append(vehicle)
    summary = {"source_rows": len(source_rows), "candidate_period_rows": len(audit),
               "candidate_status_counts": dict(sorted(Counter(r["eligibility"] for r in audit).items())),
               "eligible_by_month": {f"2019-{month:02d}": sum(len(v) for d, v in pool.items() if d.startswith(f"2019-{month:02d}")) for month in MONTHS}}
    return pool, audit, summary


def observed_cases() -> tuple[dict[str, dict], list[dict], list[dict], list[dict], dict]:
    pool, candidates, summary = observed_pool()
    cases, selected, date_audit = {}, [], []
    used = set(exclusions())
    chosen_dates = set()
    for month in MONTHS:
        wanted = list(SIZES) * 2
        selected_index = 0
        for day in range(1, calendar.monthrange(2019, month)[1] + 1):
            date = f"2019-{month:02d}-{day:02d}"
            size = wanted[selected_index]
            eligible = [v for v in pool.get(date, []) if v["physical_key"] not in used]
            row = {"month": month, "date": date, "requested_fleet_size": size,
                   "eligible_unused_sessions": len(eligible), "selected": len(eligible) >= size,
                   "reason": "first_eligible_date_for_next_size" if len(eligible) >= size else "insufficient_eligible_sessions",
                   "fleet": ""}
            date_audit.append(row)
            if len(eligible) < size:
                continue
            rank = lambda v: (digest_token(f"observed|selection|{date}|{v['physical_key']}"), v["physical_key"])
            chosen = sorted(sorted(eligible, key=rank)[:size], key=lambda v: (v["arrival_seconds"], v["departure_seconds"], v["physical_key"]))
            name = f"A2019{month:02d}_{selected_index + 1:02d}_N{size:02d}"
            row["fleet"] = name
            assert date not in chosen_dates
            chosen_dates.add(date)
            metadata = {"cohort": "unused_observed_sessions", "horizon_date": date,
                        "source": "supplied processed ACN caltech.csv", "selection_seed": SEED,
                        "profile": "constant", "constructed_battery_parameters": True,
                        "observed_fields": ["Start", "End", "Energy (delivered grid kWh)"],
                        "energy_mapping": "Etarget=E0+eta_c*observed_grid_kWh",
                        "capacity_energy_coordinate": "usable battery energy; Emin=0",
                        "battery_rating_construction": "Emax=Etarget+hash_assigned_headroom; not an observed battery rating",
                        "physical_keys": [v["physical_key"] for v in chosen],
                        "source_export_row_identifiers": [v["source_export_row_identifier"] for v in chosen]}
            cases[name] = make_model(name, chosen, "constant", metadata)
            for i, v in enumerate(chosen):
                assert v["physical_key"] not in used
                used.add(v["physical_key"])
                selected.append({"fleet": name, "N": size, "horizon_date": date, "vehicle_index": i,
                                 **v, "selection_hash": rank(v)[0]})
            selected_index += 1
            if selected_index == len(wanted):
                break
        if selected_index != 6:
            raise RuntimeError(f"Month {month} supplies {selected_index}/6 predeclared fleets; do not change the design after seeing optimization results")
    assert len(cases) == 36 and len(selected) == 504 and len(chosen_dates) == 36
    keys = [r["physical_key"] for r in selected]
    assert len(set(keys)) == 504 and not (set(keys) & exclusions())
    summary.update({"fleets": 36, "selected_sessions": 504, "distinct_dates": 36,
                    "N_counts": {n: sum(c["N"] == n for c in cases.values()) for n in SIZES},
                    "selected_keys_disjoint_with_original_162": True,
                    "all_selected_sessions_fully_contained_in_horizon": True,
                    "observed_headroom_counts_kwh": dict(sorted(Counter(r["headroom_kwh"] for r in selected).items()))})
    return cases, selected, candidates, date_audit, summary


def synthetic_cases() -> tuple[dict[str, dict], list[dict], dict]:
    cases, manifest = {}, []
    for n, pattern, replicate, surplus, headroom, profile in product(
            SIZES, ("clustered", "staggered"), (1, 2), (0, 1), ("low", "ample"), ("constant", "varying")):
        base = f"N{n:02d}_{pattern}_r{replicate}"
        name = f"S_{base}_s{surplus}_h{headroom}_p{profile}"
        surplus_order = sorted(range(n), key=lambda i: digest_token(f"synthetic|{base}|surplus_rank|{i}"))
        surplus_indices = set(surplus_order[:n // 3]) if surplus else set()
        vehicles = []
        for i in range(n):
            token = f"synthetic|{base}|{i}"
            draw = lambda field: uniform(token + "|" + field)
            pick = lambda field, options: choice(token + "|" + field, options)
            if pattern == "clustered":
                arrival, departure = 7 + 2 * draw("arrival"), 15 + 3 * draw("departure")
            else:
                arrival = 16 * draw("arrival")
                departure = arrival + 2 + 6 * draw("duration")
            a, b = float(round(arrival * 3600)), float(round(departure * 3600))
            pc = pick("Pc", (3.6, 7.2, 11.0))
            eta_c = pick("eta_c", (.90, .93, .96))
            maximum = pick("Emax", (40.0, 60.0, 80.0))
            room = (.25 + .5 * draw("headroom")) if headroom == "low" else (5 + 10 * draw("headroom"))
            target = maximum - room
            minimum = .1 * maximum
            if i in surplus_indices:
                initial = target + (.25 + .5 * draw("surplus_amount")) * room
            else:
                grid_demand = min((.3 + .6 * draw("charging_utilization")) * pc * (b - a) / 3600,
                                  (target - minimum) / eta_c)
                initial = target - eta_c * grid_demand
            vehicles.append({"arrival_seconds": a, "departure_seconds": b,
                             "Pc": pc, "Pd": pick("Pd", (2.0, 4.0, 7.2)),
                             "eta_c": eta_c, "eta_d": pick("eta_d", (.88, .92, .96)),
                             "E0": initial, "Emin": minimum, "Emax": maximum, "Etarget": target})
        metadata = {"cohort": "paired_synthetic_factorial", "base_configuration_id": base,
                    "seed": SEED, "N": n, "connection_pattern": pattern, "replicate": replicate,
                    "surplus_fraction": 1 / 3 if surplus else 0,
                    "surplus_vehicle_indices": sorted(surplus_indices), "headroom_regime": headroom,
                    "profile": profile, "all_inputs_constructed": True,
                    "paired_draws": "arrival/departure/powers/efficiencies/Emax and latent uniforms shared across the eight surplus/headroom/profile variants of this base configuration"}
        cases[name] = make_model(name, vehicles, profile, metadata)
        manifest.append({"fleet": name, "base_configuration_id": base, "N": n,
                         "connection_pattern": pattern, "replicate": replicate,
                         "surplus_fraction": metadata["surplus_fraction"], "headroom_regime": headroom,
                         "profile": profile, "vehicle_count_with_initial_surplus": sum(v["E0"] > v["Etarget"] + 1e-10 for v in vehicles)})
    assert len(cases) == 96 and len({r["base_configuration_id"] for r in manifest}) == 12
    return cases, manifest, {"scenario_count": 96, "base_configuration_count": 12,
                             "variants_per_base_configuration": 8,
                             "independent_replications_claimed": False}


def build_payloads() -> tuple[dict[str, bytes], dict]:
    observed, selected, candidates, dates, obs_summary = observed_cases()
    synthetic, synthetic_manifest, syn_summary = synthetic_cases()
    payloads = {f"observed_cases/{name}.json": json_bytes(case) for name, case in observed.items()}
    payloads.update({f"synthetic_cases/{name}.json": json_bytes(case) for name, case in synthetic.items()})
    payloads.update({"observed_selected_sessions.csv": csv_bytes(selected),
                     "observed_candidate_audit.csv": csv_bytes(candidates),
                     "observed_date_selection_audit.csv": csv_bytes(dates),
                     "synthetic_factorial_manifest.csv": csv_bytes(synthetic_manifest)})
    summary = {"observed": obs_summary, "synthetic": syn_summary,
               "all_model_cases": 132, "all_integer_q_values_per_case": "0,...,N",
               "nominal_case_q_lp_calls_planned": sum(c["N"] + 1 for c in [*observed.values(), *synthetic.values()]),
               "input_validation": "positive durations, exact availability, energy bounds, full-power individual deliverability; no capacity optimization used"}
    payloads["input_summary.json"] = json_bytes(summary)
    return payloads, summary


def specification() -> dict:
    return {"seed": SEED,
            "observed_period": ["2019-04-01", "2019-10-01"],
            "observed_design": "Each April-September2019 month: first chronological eligible dates for N sequence [6,12,24,6,12,24]; one fleet per date; SHA256-ranked sessions; 36 fleets and504 distinct sessions",
            "observed_eligibility": "Finite observed grid energy>=2kWh; full session within one24h local day; positive duration; deduplicated physical key; exclude original144 and old18; Egrid<=hash-assigned Pc*actual hours. No full delivered energy assigned to a clipped window.",
            "timestamp_convention": "Naive processed-export Start/End interpreted in America/Los_Angeles; UTC duration arithmetic",
            "observed_parameters": {"Pc_kw": [3.6, 7.2, 11], "Pd_kw": [2, 4, 7.2],
                                    "eta_c": [.90, .93, .96], "eta_d": [.88, .92, .96],
                                    "E0_kwh": [15, 25, 35], "headroom_kwh": [0, .5, 5, 15],
                                    "Emin_kwh": 0, "Etarget": "E0+eta_c*observed grid kWh",
                                    "Emax": "Etarget+headroom; demand-conditioned assumed battery rating",
                                    "L_kw": 10, "Ulo": 1, "Uhi": 1},
            "synthetic_design": {"N": [6, 12, 24], "surplus_fraction": [0, 1 / 3],
                                 "headroom": ["low .25-.75kWh", "ample5-15kWh"],
                                 "connection_pattern": ["clustered arrivals7-9h/departures15-18h", "staggered arrivals0-16h/durations2-8h"],
                                 "profile": ["constant", "varying"], "replicates": 2,
                                 "count": 96, "paired_base_configurations": 12,
                                 "Emax_kwh": [40, 60, 80], "Emin": ".1*Emax", "Etarget": "Emax-headroom",
                                 "charging_request": "E0=Etarget-eta_c*g; g=min((.3+.6u)*Pc*available_hours,(Etarget-Emin)/eta_c)",
                                 "surplus_request": "E0=Etarget+(.25+.5u)*headroom",
                                 "powers_and_efficiencies": "Same finite choice sets as observed-session cohort, keyed to synthetic base configuration and vehicle"},
            "varying_profile": {"hour": "floor(event-interval midpoint in hours)",
                                "L_kw": "6+4*exp(-((hour-16)/4)^2)", "Ulo": .9,
                                "Uhi": "1.1+.3*sin(pi*(hour-7)/24)^2"},
            "time_grid": "Union of all exact arrival/departure seconds and hourly boundaries; positive interval durations; binary availability; no outward rounding",
            "q_values": "All integer q=0,...,N in every case; common no-discharge guard at every q",
            "retention_policy": "Retain every generated case, including background-dominated or optimizer-infeasible cases; no selection by capacity, saving, certificate success or significance",
            "interpretation": "Disjoint future-session contexts from the supplied processed export, not a new site or held-out draw from a specified population. Synthetic96 are paired mechanism scenarios on12 base configurations. No field-V2G prevalence or measured battery-state claim.",
            "source_provenance_limit": "The previous upstream ACN raw-data comparison covered January-March2019 only; April-September2019 data are pinned processed-export records, not independently re-acquired or newly verified raw sessions."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="generate inputs and freeze their hashes before optimization")
    parser.add_argument("--check", action="store_true", help="reconstruct and verify the existing frozen inputs without writing")
    args = parser.parse_args()
    if args.freeze == args.check:
        parser.error("Choose exactly one of --freeze or --check")
    payloads, summary = build_payloads()
    source_hashes = {name: sha256(SOURCE / name) for name in ("caltech.csv", "frozen_selected_sessions.csv", "original_excluded_sessions.csv")}
    content = {"generator_sha256": sha256(Path(__file__)), "source_hashes": source_hashes,
               "design": specification(), "input_sha256": {name: sha256_bytes(value) for name, value in sorted(payloads.items())},
               "input_summary": summary}
    # JSON object keys are strings after loading (e.g. fleet sizes/headroom).
    # Normalize the comparison representation without altering any input bytes.
    content = json.loads(json_bytes(content))
    if PROTOCOL.exists():
        retained = json.loads(PROTOCOL.read_text())
        for key, value in content.items():
            if retained[key] != value:
                raise RuntimeError(f"Frozen {key} changed; retain the existing protocol and document an amendment before any altered experiment")
        for name, value in payloads.items():
            path = OUTPUT / name
            if not path.is_file() or path.read_bytes() != value:
                raise RuntimeError(f"Frozen input differs or is missing: {name}")
        print(json.dumps({"status": "PASS", "mode": "reconstruction check", "summary": summary}, indent=2))
        return
    if args.check:
        raise FileNotFoundError("No frozen data protocol exists")
    if OUTPUT.exists() and any(OUTPUT.rglob("*")):
        raise RuntimeError("Input directory is nonempty without a protocol; inspect it before generating")
    for name, value in payloads.items():
        path = OUTPUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    protocol = {"study": "Round10 independent context and mechanism validation",
                "frozen_utc_before_any_optimization": datetime.now(UTC).isoformat(),
                "no_optimization_results_read_during_design": True, **content}
    PROTOCOL.write_bytes(json_bytes(protocol))
    print(json.dumps({"status": "FROZEN", "protocol": str(PROTOCOL),
                      "frozen_utc": protocol["frozen_utc_before_any_optimization"],
                      "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
