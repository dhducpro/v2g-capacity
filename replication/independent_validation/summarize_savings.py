"""Summarize the post hoc comparisons in Section 3.1."""
import csv
import json
from pathlib import Path

study = Path(__file__).resolve().parent
output = study.parents[1] / "outputs" / "savings_comparisons.csv"
rows = []

for path in sorted((study / "results" / "cases").glob("*.json")):
    result = json.loads(path.read_text(encoding="utf-8"))
    fleet = json.loads((study / result["input_relative_path"]).read_text(encoding="utf-8"))
    summary = result["summary"]
    donors = [availability for availability, power, upper, target in
              zip(fleet["availability"], fleet["Pd"], fleet["Emax"], fleet["Etarget"])
              if power > 0 and upper > target]
    counts = [sum(donor[t] > 0 for donor in donors) for t in range(fleet["T"])]
    headroom = min(max(counts[t] for t in cut["slots"])
                   for cut in result["cut_extraction"]["cuts"])
    rows.append({
        "cohort": summary["cohort"], "case": summary["case"],
        "q_occupancy": summary["q_occupancy"], "q_headroom": headroom,
        "q_comparator": min(headroom, summary["q_occupancy"]),
        "q_certificate": summary["q_certificate"],
        "reservation_kw": summary["M_reservation_kw"],
        "saving_q0_kw": summary["nominal_saving_kw"],
        "saving_q1_kw": summary["one_withdrawal_saving_kw"],
    })

output.parent.mkdir(exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for cohort in ("bridge", "temporal", "mechanism"):
    group = [row for row in rows if row["cohort"] == cohort]
    improved = sum(row["q_certificate"] < row["q_comparator"] for row in group)
    tied = sum(row["q_certificate"] == row["q_comparator"] for row in group)
    print(f"{cohort}: lower threshold in {improved}/{len(group)}, equal in {tied}")

temporal = [row for row in rows if row["cohort"] == "temporal"]
for field in ("saving_q0_kw", "saving_q1_kw"):
    print(field)
    for cutoff in (1e-6, 0.1, 1.0):
        print(f"  above {cutoff:g} kW: {sum(row[field] > cutoff for row in temporal)}")
    print("  above 1% of reservation:", sum(row[field] > 0.01 * row["reservation_kw"]
                                           for row in temporal))
