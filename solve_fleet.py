"""Solve one fleet for a given discharge withdrawal budget."""

import argparse
import json
from pathlib import Path

from v2g_capacity import (
    compute_certificate,
    reservation_cut,
    solve_capacity,
    solve_reservation,
)


def summarize(model, q):
    reservation, _ = solve_reservation(model)
    protected, _ = solve_capacity(model, q)
    summary = {
        "fleet": model.get("name", "unnamed"),
        "q": q,
        "reservation_status": reservation["status"],
        "reservation_kw": reservation["M_kw"],
        "protected_status": protected["status"],
        "protected_kw": protected["M_kw"],
        "saving_kw": None,
        "bound_kw": None,
    }
    if reservation["status"] == protected["status"] == "optimal":
        summary["saving_kw"] = reservation["M_kw"] - protected["M_kw"]
        cuts = reservation_cut(model, reservation["M_kw"])
        summary["cut_status"] = cuts["status"]
        summary["bound_kw"] = min(
            (compute_certificate(model, cut, q)["bound_kw"] for cut in cuts["cuts"]),
            default=None,
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fleet", type=Path, help="Fleet model in JSON format")
    parser.add_argument(
        "--q", type=int, required=True, help="Maximum number of withdrawn discharge plans"
    )
    args = parser.parse_args()
    try:
        model = json.loads(args.fleet.read_text(encoding="utf-8"))
        result = summarize(model, args.q)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
