"""Build the input-derived W206 explanation and audit W203 arithmetic.

No optimization or new sampling is performed. The retained critical cut is
read from the structural audit, numerical capacities from the existing all-q
LP records, and every physical quantity from the frozen case JSON files.
Run from any directory: python replication/build_capacity_mechanism.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "replication" / "capacity_checks"
CASES = CHECKS / "inputs" / "withdrawal_budget_cases"
OUT = ROOT / "outputs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def cut_quantities(case, slots):
    a = np.asarray(case["availability"], float)
    dt = float(case["dt"])
    pc = a * np.asarray(case["Pc"], float)[:, None]
    pd = a * np.asarray(case["Pd"], float)[:, None]
    eta_c, eta_d = (np.asarray(case[k], float) for k in ("eta_c", "eta_d"))
    initial, target, upper = (np.asarray(case[k], float) for k in ("E0", "Etarget", "Emax"))
    h = (target - initial) / (eta_c * dt)
    outside = pc.sum(axis=1) - pc[:, slots].sum(axis=1)
    required = np.maximum(h - outside, 0)
    outside_slack = np.maximum(outside - h, 0)
    guard = eta_d * (upper - target) / dt
    recharge = eta_c * eta_d * outside_slack
    budget = np.minimum(guard, recharge)
    donors = (budget > 1e-9)[:, None] & (pd[:, slots] > 1e-9)
    return dict(
        availability=a,
        dt=dt,
        h=h,
        outside=outside,
        required=required,
        outside_slack=outside_slack,
        guard=guard,
        recharge=recharge,
        budget=budget,
        donors=donors,
        pd=pd,
    )


def main():
    OUT.mkdir(exist_ok=True)
    cut_path = CHECKS / "critical_cut_details.csv"
    frontier_path = CHECKS / "full_frontier_independent_lp.csv"
    cuts = read_csv(cut_path)
    frontier = read_csv(frontier_path)
    inputs = [cut_path, frontier_path]
    cases = {cid: json.loads((CASES / f"{cid}.json").read_text()) for cid in ("W206", "W203")}
    data, rows = {}, []
    for cid, case in cases.items():
        source = next(row for row in cuts if row["fleet"] == cid and row["q"] == "0")
        slots = np.array([int(v) - 1 for v in source["slots_1based"].split()])
        values = cut_quantities(case, slots)
        data[cid] = {"slots": slots, **values}
        for i in range(case["N"]):
            available = np.flatnonzero(values["availability"][i]) + 1
            assert np.array_equal(available, np.arange(available[0], available[-1] + 1))
            row = {
                "fleet": cid,
                "vehicle": i + 1,
                "available_slots_1based": " ".join(map(str, available)),
                "cut_slots_1based": source["slots_1based"],
            }
            for name in (
                "h",
                "outside",
                "required",
                "outside_slack",
                "guard",
                "recharge",
                "budget",
            ):
                row[name + "_kwh"] = float(values[name][i] * values["dt"])
            rows.append(row)
    with (OUT / "capacity_mechanism_inputs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    max_battery_error, inflated, battery_max = 0.0, 0, 0.0
    for path in sorted(CASES.glob("*.json")):
        inputs.append(path)
        case = json.loads(path.read_text())
        delivered = np.array(
            [float(key.rsplit("|", 1)[1]) for key in case["metadata"]["selected_physical_keys"]]
        )
        initial_size = np.maximum(55.0, np.minimum(100.0, delivered / 0.45 + 20.0))
        initial = 0.4 * initial_size
        target = initial + delivered
        upper = np.maximum(initial_size, target / 0.92)
        for key, expected in (
            ("E0", initial),
            ("Etarget", target),
            ("Emax", upper),
            ("Emin", 0.1 * upper),
        ):
            max_battery_error = max(
                max_battery_error, float(np.max(np.abs(np.asarray(case[key]) - expected)))
            )
        inflated += int(np.count_nonzero(upper > 55.0 + 1e-9))
        battery_max = max(battery_max, float(upper.max()))
    assert max_battery_error < 1e-10

    w = data["W206"]
    assert np.array_equal(w["slots"], [13, 14]) and w["dt"] == 1.0
    assert np.array_equal(w["donors"].sum(axis=0), [4, 4])
    assert np.all(w["budget"][:4] >= 8.0)
    assert np.all(w["pd"][:4, :][:, w["slots"]] == 4.0)
    mandatory = float(w["required"].sum() * w["dt"])
    cut_weight = len(w["slots"]) / 1.5
    reservation = 10.0 + float(w["required"].sum()) / cut_weight
    retained = {int(row["q"]): float(row["M_kw"]) for row in frontier if row["fleet"] == "W206"}
    assert abs(reservation - retained[6]) < 1e-10
    # Each donor can supply 4 kW in both slots of the relaxed credit model.
    residual_credit = np.array([8.0 * max(4 - q, 0) for q in range(7)])
    for row in cuts:
        if row["fleet"] == "W206":
            q = int(row["q"])
            assert abs(residual_credit[q] / cut_weight - float(row["upper_bound_saving_kw"])) < 1e-8

    table = [
        r"\begin{tabular}{rlrrrr}",
        r"\toprule",
        r"EV & Available slots & $h_i\Delta t$ & $P_i^{\rm out}\Delta t$ & $[h_i-P_i^{\rm out}]_+\Delta t$ & $e_i\Delta t$\\",
        r"\midrule",
    ]
    for row in (r for r in rows if r["fleet"] == "W206"):
        slots = row["available_slots_1based"].split()
        if row["vehicle"] == 5:
            table.append(r"\midrule")
        table.append(
            f"{row['vehicle']} & {slots[0]}--{slots[-1]} & "
            + " & ".join(
                f"{row[k]:.3f}" for k in ("h_kwh", "outside_kwh", "required_kwh", "budget_kwh")
            )
            + r"\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "capacity_w206_inputs_table.tex").write_text("\n".join(table) + "\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(
        1, 2, figsize=(7.3, 3.15), layout="constrained", gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    donor_color, request_color = "#168a79", "#dd7b22"
    ax[0].axvspan(13.5, 15.5, facecolor="#eeeeee", edgecolor="#777777", lw=1.0, zorder=0)
    for i, a in enumerate(w["availability"]):
        slots = np.flatnonzero(a) + 1
        ax[0].barh(
            i + 1,
            len(slots),
            left=slots[0] - 0.5,
            height=0.56,
            color=donor_color if i < 4 else request_color,
            edgecolor="white",
            lw=0.7,
            zorder=2,
        )
    for x in (13.5, 15.5):
        ax[0].axvline(x, color="#555555", ls="--", lw=0.9, zorder=3)
    ax[0].set(
        xlim=(8.4, 20.6),
        ylim=(6.65, 0.2),
        xticks=np.arange(9, 21),
        yticks=np.arange(1, 7),
        yticklabels=[f"EV {i}" for i in range(1, 7)],
        xlabel="Modeled hourly slot (one-based)",
        title="(a) Availability and the critical cut",
    )
    ax[0].text(
        14.5,
        0.43,
        "S = {14, 15}",
        ha="center",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
    )
    ax[0].legend(
        handles=[
            Patch(color=donor_color, label="Positive donor budget"),
            Patch(color=request_color, label="Forced charging in S"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
        ncol=1,
        fontsize=8,
    )

    qvals = np.arange(7)
    ax[1].barh(qvals, residual_credit, height=0.55, color=donor_color, alpha=0.85)
    for q, value in enumerate(residual_credit):
        ax[1].text(value + 0.6, q, f"{value:.0f}", va="center", fontsize=8)
    ax[1].axvline(mandatory, color=request_color, ls="--", lw=1.4)
    ax[1].text(
        mandatory + 0.7,
        5.7,
        f"Forced charging\n{mandatory:.3f} kWh",
        color="#9a4c10",
        fontsize=8,
        va="center",
    )
    ax[1].set(
        xlim=(0, 35),
        ylim=(6.65, -0.65),
        yticks=qvals,
        ylabel=r"Protected withdrawals $q$",
        xlabel=r"Cut credit $R_q(S)\Delta t$ (kWh)",
        title="(b) Credit allowed by the cut bound",
    )
    ax[1].grid(axis="x", alpha=0.17)
    fig.savefig(OUT / "capacity_w206_mechanism.pdf")
    fig.savefig(OUT / "capacity_w206_mechanism.png", dpi=190)
    plt.close(fig)

    # With EV4 withdrawn, only EV5 and EV6 can serve the late slots.
    case = cases["W203"]
    late = np.array([19, 20])
    a = np.asarray(case["availability"], float)
    assert np.array_equal(np.flatnonzero(a[:, late].any(axis=1)), [3, 4, 5])
    pc = np.asarray(case["Pc"], float)[:, None] * a
    h = data["W203"]["h"]
    before_late_max = float((pc[[4, 5]].sum() - pc[[4, 5]][:, late].sum()) * case["dt"])
    late_required = float(h[[4, 5]].sum() * case["dt"] - before_late_max)
    late_bound = 10.0 + 1.5 * late_required / (len(late) * case["dt"])
    w203_values = {int(row["q"]): float(row["M_kw"]) for row in frontier if row["fleet"] == "W203"}
    assert all(abs(late_bound - w203_values[q]) < 1e-10 for q in (1, 2))

    examples = []
    n, q, k, demand = 4, 1, 0.81, 4.0
    for total_rating in (3.0, 1.5):
        b = total_rating / n
        y = min(b, k * demand / (n + k * (n - q)))
        slot1, slot2 = demand - (n - q) * y, n * y / k
        examples.append(
            {
                "n": n,
                "q": q,
                "k": k,
                "D_kw": demand,
                "B_kw": total_rating,
                "per_donor_discharge_kw": y,
                "per_donor_recharge_kw": y / k,
                "slot1_demand_above_L_kw": slot1,
                "slot2_demand_above_L_kw": slot2,
                "saving_kw": demand - max(slot1, slot2),
            }
        )
    audit = {
        "scope": "Input-derived explanation of retained cases; no new optimization or sampling",
        "optimizer_calls": 0,
        "source_sha256": digest(Path(__file__)),
        "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "battery_construction": {
            "checked_EVs": 144,
            "max_error_kwh": max_battery_error,
            "Emax_above_55_count": inflated,
            "max_Emax_kwh": battery_max,
        },
        "W206": {
            "cut_slots_1based": [14, 15],
            "mandatory_grid_energy_kwh": mandatory,
            "reservation_capacity_kw": reservation,
            "positive_donors": [1, 2, 3, 4],
            "donor_budget_binding": "common battery guard for all four useful donors",
            "cut_credit_kwh_q0_to_q6": residual_credit.tolist(),
            "retained_capacity_kw": retained,
        },
        "W203": {
            "late_slots_1based": [20, 21],
            "withdrawn_donor": 4,
            "EV5_EV6_grid_requirement_kwh": float(h[[4, 5]].sum()),
            "max_outside_late_grid_charge_kwh": before_late_max,
            "late_required_grid_energy_kwh": late_required,
            "lower_bound_for_q_at_least_1_kw": late_bound,
            "attained_at_q": [1, 2],
        },
        "analytic_schedule_illustrations": examples,
        "all_arithmetic_checks_passed": True,
    }
    (OUT / "capacity_mechanism_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                "W206": audit["W206"],
                "W203": audit["W203"],
                "battery_construction": audit["battery_construction"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
