"""Solve the charging cases used to illustrate the hourly bottlenecks."""
import csv
import json
from pathlib import Path

from full_lp_checks import solve_capacity

inputs = Path(__file__).resolve().parent / "inputs"
rows = []
for path in sorted((inputs / "withdrawal_budget_cases").glob("*.json")):
    case = json.loads(path.read_bytes())
    values = {q: solve_capacity(case, q)["M_kw"] for q in (0, 1, 2, 6)}
    row = {"fleet": path.stem}
    row.update({f"M{q}_kw": value for q, value in values.items()})
    row.update({f"M6_minus_M{q}_kw": values[6] - value for q, value in values.items()})
    rows.append(row)
with (inputs / "withdrawal_budget_capacity_contexts.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
