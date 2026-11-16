"""Format compact manuscript tables from existing verified results; no solves."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
source = ROOT / "replication/capacity_checks/full_frontier_independent_lp.csv"
rows = list(csv.DictReader(source.open()))
assert len(rows) == 168
fleets = {r["fleet"] for r in rows}
assert len(fleets) == 24
lines = [
    r"\begin{tabular}{@{}rrrrr@{}}",
    r"\toprule",
    r"$q$ & \shortstack{Positive\\fleets} & \shortstack{Certified\\zero} & \shortstack{Mean\\(kW)} & \shortstack{Maximum\\(kW)}\\",
    r"\midrule",
]
summary = []
for q in range(7):
    selected = [r for r in rows if int(r["q"]) == q]
    assert len(selected) == 24
    values = [max(0.0, float(r["saving_vs_reservation_kw"])) for r in selected]
    values = [v if v >= 1e-6 else 0.0 for v in values]
    positive = sum(v > 1e-6 for v in values)
    certified = sum(q >= int(r["cut_collapse_q"]) for r in selected)
    mean, maximum = sum(values) / 24, max(values)
    lines.append(f"{q} & {positive} & {certified} & {mean:.4f} & {maximum:.4f}" + r"\\")
    summary.append(
        {
            "q": q,
            "positive": positive,
            "certified_zero": certified,
            "mean_saving_kw": mean,
            "maximum_saving_kw": maximum,
        }
    )
assert [r["positive"] for r in summary] == [13, 3, 2, 2, 0, 0, 0]
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "publication_frontier_table.tex").write_text("\n".join(lines) + "\n")

mechanism = OUT / "capacity_mechanism_inputs.csv"
mech_rows = [r for r in csv.DictReader(mechanism.open()) if r["fleet"] == "W206"]
assert len(mech_rows) == 6 and all(r["fleet"] == "W206" for r in mech_rows)
lines = [
    r"\begin{tabular}{@{}rlrrrr@{}}",
    r"\toprule",
    r"EV & Available slots & \shortstack{Requirement\\$h_i$ (kWh)} & \shortstack{Outside charging\\$P_i^{\rm out}$ (kWh)} & \shortstack{Forced into $S$\\$[h_i-P_i^{\rm out}]_+$ (kWh)} & \shortstack{Donor allowance\\$e_i$ (kWh)}\\",
    r"\midrule",
]
for row in mech_rows:
    if int(row["vehicle"]) == 5:
        lines.append(r"\midrule")
    slots = [int(x) for x in row["available_slots_1based"].split()]
    assert slots == list(range(slots[0], slots[-1] + 1))
    energies = [float(row[key]) for key in ("h_kwh", "outside_kwh", "required_kwh", "budget_kwh")]
    lines.append(
        f"{row['vehicle']} & {slots[0]}--{slots[-1]} & "
        + " & ".join(f"{v:.3f}" for v in energies)
        + r"\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "publication_w206_inputs_table.tex").write_text("\n".join(lines) + "\n")
audit = {
    "source_frontier_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source_mechanism_sha256": hashlib.sha256(mechanism.read_bytes()).hexdigest(),
    "frontier_rows": summary,
    "w206_rows": len(mech_rows),
    "omitted_from_compact_frontier": "Zero count is 24 minus positive; median is retained in the full table and data.",
    "new_optimizer_calls": 0,
}
(OUT / "publication_table_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("Publication tables rebuilt from 168 saved capacities and six W206 input rows.")
