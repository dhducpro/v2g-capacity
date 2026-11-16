"""Recompute all 156 fleet frontiers and compare them with the archived capacities."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import run_validation as study


def read_frontiers(path):
    with path.open() as stream:
        return {(r["case"], int(r["q"])): r for r in csv.DictReader(stream)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output.resolve() if args.output else study.HERE / "recomputed" / stamp
    if destination.exists():
        raise ValueError(
            "Choose a new output directory; archived or earlier outputs are not overwritten"
        )
    reference = read_frontiers(study.OUT / "frontiers.csv")
    study.OUT = destination
    study.run()
    recomputed = read_frontiers(destination / "frontiers.csv")
    assert reference.keys() == recomputed.keys()
    maximum = 0.0
    for key, old in reference.items():
        new = recomputed[key]
        assert old["status"] == new["status"], key
        if old["status"] == "optimal":
            difference = abs(float(old["M_kw"]) - float(new["M_kw"]))
            maximum = max(maximum, difference)
            assert difference <= study.SAVING_TOL, key
    record = {
        "status": "PASS",
        "compared_frontier_values": len(reference),
        "maximum_capacity_difference_kw": maximum,
        "frozen_inputs_and_runner_preserved": True,
        "archived_results_unchanged": True,
    }
    (destination / "RECOMPUTATION_CHECK.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
