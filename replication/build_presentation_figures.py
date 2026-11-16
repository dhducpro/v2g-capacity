"""Build the five journal figures from frozen records and proved formulas.

Publication widths are 178 mm for two columns and 86 mm for one column.
The plotted values and source digests are in outputs/presentation_figure_audit.json.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHECKS = ROOT / "replication" / "capacity_checks"
SENSITIVITY = ROOT / "replication" / "sensitivity_checks"
WIDTH = 178 / 25.4
TOL = 1e-6
BLUE = "#216B86"
ORANGE = "#B66032"
TEAL = "#168A79"
GRAY = "#687078"
LIGHT = "#F0F2F3"
INK = "#252A2D"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def displayed_saving(value: str | float) -> float:
    number = float(value)
    assert number >= -TOL, f"Unexpected negative saving: {number}"
    return 0.0 if abs(number) <= TOL else number


def finish(fig: plt.Figure, stem: str) -> None:
    fig.savefig(
        OUT / f"{stem}.pdf",
        metadata={
            "Title": stem.replace("presentation_", "").replace("_", " "),
            "Subject": "Scientific illustration from frozen records or analytical formulas",
            "Creator": "build_presentation_figures.py",
        },
    )
    fig.savefig(OUT / f"{stem}.png", dpi=240)
    plt.close(fig)


def panel(ax: plt.Axes, label: str, name: str = "") -> None:
    text = label if not name else f"{label}  {name}"
    ax.set_title(text, loc="left", pad=7, fontsize=9, fontweight="bold")


def tidy(ax: plt.Axes, *, axis: str = "y") -> None:
    ax.grid(axis=axis, color="#DCE1E3", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def make_contract() -> dict:
    """Illustrate a fixed nominal plan and one cancellation, not a new run."""
    n, demand, total_rating = 4, 4.0, 3.0
    eta_c = eta_d = 0.9
    k, dt = eta_c * eta_d, 1.0
    b = total_rating / n
    y = min(b, k * demand / (n + k * n))
    recharge = y / k
    initial = target = b / eta_d
    upper = 2 * b / eta_d
    nominal_import = np.array([demand - n * y, n * recharge])
    cancelled_import = np.array([demand - (n - 1) * y, n * recharge])
    planned_energy = np.array([initial, initial - dt * y / eta_d, target])
    cancelled_energy = np.array([initial, initial, initial + eta_c * dt * recharge])
    assert np.ptp(nominal_import) < 1e-12
    assert abs(cancelled_import[0] - nominal_import[0] - y) < 1e-12
    assert cancelled_import[1] == nominal_import[1]
    assert np.all(planned_energy >= 0) and cancelled_energy.max() <= upper
    assert abs(cancelled_energy[-1] - (b + y) / eta_d) < 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.2))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.245, wspace=0.33)
    x, width = np.arange(2), 0.32
    for offset, values, color, hatch in (
        (-width / 2, nominal_import, BLUE, None),
        (width / 2, cancelled_import, ORANGE, "//"),
    ):
        bars = axes[0].bar(
            x + offset,
            values,
            width=width,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.055,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].axhline(nominal_import[0], color=GRAY, linewidth=1, linestyle=":", zorder=2)
    axes[0].text(
        1.49,
        nominal_import[0] - 0.075,
        r"$M_0^\star$",
        ha="right",
        va="top",
        color=GRAY,
        fontsize=9,
    )
    axes[0].set(
        xlim=(-0.55, 1.55),
        ylim=(0, 3.18),
        xticks=x,
        xticklabels=["1: support", "2: recharge"],
        xlabel="One-hour slot",
        ylabel="Station net import (kW)",
    )
    panel(axes[0], "(a)")
    tidy(axes[0])

    times = np.arange(3)
    axes[1].plot(
        times, planned_energy, color=BLUE, marker="o", linewidth=1.65, markersize=4, zorder=4
    )
    axes[1].plot(
        times, cancelled_energy, color=ORANGE, marker="s", linewidth=1.65, markersize=4, zorder=3
    )
    axes[1].axhline(upper, color=GRAY, linestyle="--", linewidth=1)
    axes[1].text(
        1.99,
        upper + 0.04,
        r"Upper limit $E_i^{\max}$",
        ha="right",
        va="bottom",
        color=GRAY,
        fontsize=8,
    )
    axes[1].axhline(target, color=GRAY, linestyle=":", linewidth=0.9)
    axes[1].text(
        0.03,
        target + 0.04,
        r"$E_{0,i}=E_i^{\rm tar}$",
        ha="left",
        va="bottom",
        color=GRAY,
        fontsize=8,
    )
    axes[1].set(
        xlim=(-0.07, 2.07),
        ylim=(0, 1.87),
        xticks=times,
        xlabel="Elapsed time (h)",
        ylabel="One donor's energy (kWh)",
    )
    panel(axes[1], "(b)")
    tidy(axes[1])
    fig.legend(
        handles=[
            Patch(facecolor=BLUE, label="Planned operation"),
            Patch(
                facecolor=ORANGE,
                hatch="//",
                edgecolor="white",
                label="One cancellation, fixed charging",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.6,
        labelspacing=0.5,
        columnspacing=2,
    )
    finish(fig, "presentation_contract")
    return {
        "basis": "Analytical two-slot construction; nominal q=0 optimal plan, then cancellation under that identical plan.",
        "not_a_reoptimized_q1_frontier": True,
        "n_donors": n,
        "D_kw": demand,
        "B_kw": total_rating,
        "L_kw": 0,
        "u": 1,
        "eta_c": eta_c,
        "eta_d": eta_d,
        "dt_hours": dt,
        "per_donor_discharge_kw": y,
        "per_donor_fixed_charging_kw": recharge,
        "nominal_import_kw": nominal_import.tolist(),
        "same_plan_one_cancellation_import_kw": cancelled_import.tolist(),
        "donor_E0_and_target_kwh": initial,
        "donor_Emax_kwh": upper,
        "time_hours": times.tolist(),
        "planned_donor_energy_kwh": planned_energy.tolist(),
        "cancelled_donor_energy_kwh": cancelled_energy.tolist(),
    }


def make_frontier(rows: list[dict]) -> dict:
    fleets = sorted({row["fleet"] for row in rows})
    assert len(rows) == 168 and len(fleets) == 24
    series = {(row["fleet"], int(row["q"])): row for row in rows}
    assert len(series) == 168
    saving = np.array(
        [
            [displayed_saving(series[fleet, q]["saving_vs_reservation_kw"]) for q in range(7)]
            for fleet in fleets
        ]
    )
    capacities = np.array([[float(series[fleet, q]["M_kw"]) for q in range(7)] for fleet in fleets])
    np.testing.assert_allclose(saving, capacities[:, [6]] - capacities, atol=TOL, rtol=0)
    counts = (saving > TOL).sum(axis=0).tolist()
    assert counts == [13, 3, 2, 2, 0, 0, 0]
    assert np.max(np.diff(saving, axis=1)) <= TOL

    fig = plt.figure(figsize=(WIDTH, 4.45))
    ax0 = fig.add_axes([0.095, 0.13, 0.315, 0.805])
    cax = fig.add_axes([0.425, 0.13, 0.017, 0.805])
    ax1 = fig.add_axes([0.565, 0.13, 0.42, 0.805])
    cmap = LinearSegmentedColormap.from_list("capacity_blue", ["#C8E0E8", BLUE])
    cmap.set_bad(LIGHT)
    image = ax0.pcolormesh(
        np.arange(-0.5, 7.5, 1),
        np.arange(-0.5, 24.5, 1),
        np.ma.masked_equal(saving, 0),
        cmap=cmap,
        norm=Normalize(0, 6),
        shading="flat",
        zorder=1,
    )
    ax0.set(
        xticks=range(7),
        yticks=range(24),
        yticklabels=fleets,
        xlim=(-0.5, 6.5),
        ylim=(23.5, -0.5),
        xlabel=r"Protected withdrawals $q$",
    )
    ax0.tick_params(axis="y", length=0, labelsize=8)
    ax0.tick_params(axis="x", length=0)
    for divider in (7.5, 15.5):
        ax0.axhline(divider, color="white", linewidth=1.5)
    ax0.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax0.set_yticks(np.arange(-0.5, 24, 1), minor=True)
    ax0.grid(which="minor", color="white", linewidth=0.45)
    ax0.tick_params(which="minor", left=False, bottom=False)
    for spine in ax0.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(image, cax=cax, ticks=[0, 2, 4, 6])
    cb.set_label("Capacity saving (kW)", labelpad=5)
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    panel(ax0, "(a)")
    ax0.legend(
        handles=[
            Patch(
                facecolor=LIGHT,
                edgecolor="#C3C9CC",
                linewidth=0.5,
                label=r"Zero ($\leq 10^{-6}$ kW)",
            )
        ],
        loc="upper left",
        bbox_to_anchor=(-0.01, -0.105),
        borderaxespad=0,
        frameon=False,
        fontsize=8,
        handlelength=1.2,
    )

    selected = [("W104", BLUE, "o"), ("W203", ORANGE, "s"), ("W206", TEAL, "^")]
    for fleet, color, marker in selected:
        ax1.plot(
            range(7),
            saving[fleets.index(fleet)],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.65,
            label=fleet,
            zorder=3,
        )
    ax1.set(
        xlabel=r"Protected withdrawals $q$",
        ylabel="Capacity saving (kW)",
        xticks=range(7),
        xlim=(-0.12, 6.12),
        ylim=(-0.1, 6.2),
    )
    ax1.legend(frameon=False, loc="upper right", handlelength=2, labelspacing=0.5)
    tidy(ax1)
    panel(ax1, "(b)")
    finish(fig, "presentation_frontier")
    return {
        "fleets_in_order": fleets,
        "q": list(range(7)),
        "capacity_saving_kw": saving.tolist(),
        "capacity_kw": capacities.tolist(),
        "positive_saving_counts": counts,
        "zero_saving_cases": int((saving == 0).sum()),
        "infeasible_cases": 0,
        "selected_fleets": [value[0] for value in selected],
    }


def make_w206(case: dict, cuts: list[dict], frontier: list[dict]) -> dict:
    source = next(row for row in cuts if row["fleet"] == "W206" and row["q"] == "0")
    slots = np.array([int(value) - 1 for value in source["slots_1based"].split()])
    availability = np.asarray(case["availability"], float)
    dt = float(case["dt"])
    pc = availability * np.asarray(case["Pc"], float)[:, None]
    pd = availability * np.asarray(case["Pd"], float)[:, None]
    eta_c = np.asarray(case["eta_c"], float)
    eta_d = np.asarray(case["eta_d"], float)
    initial = np.asarray(case["E0"], float)
    target = np.asarray(case["Etarget"], float)
    upper = np.asarray(case["Emax"], float)
    h = (target - initial) / (eta_c * dt)
    outside = pc.sum(axis=1) - pc[:, slots].sum(axis=1)
    required = np.maximum(h - outside, 0) * dt
    guard = eta_d * (upper - target)
    recharge = eta_c * eta_d * np.maximum(outside - h, 0) * dt
    budget = np.minimum(guard, recharge)
    donors = (budget > 1e-9)[:, None] & (pd[:, slots] > 1e-9)
    assert np.array_equal(slots, [13, 14]) and dt == 1
    assert np.array_equal(donors.sum(axis=0), [4, 4])
    assert np.all(budget[:4] >= 8) and np.all(pd[:4, :][:, slots] == 4)
    q = np.arange(7)
    credit = 8 * np.maximum(4 - q, 0)
    cut_weight = len(slots) / 1.5
    for row in cuts:
        if row["fleet"] == "W206":
            np.testing.assert_allclose(
                credit[int(row["q"])] / cut_weight,
                float(row["upper_bound_saving_kw"]),
                rtol=0,
                atol=1e-8,
            )
    reservation = 10 + required.sum() / cut_weight
    retained_reservation = float(
        next(row["M_kw"] for row in frontier if row["fleet"] == "W206" and row["q"] == "6")
    )
    assert abs(reservation - retained_reservation) < 1e-10

    fig, ax = plt.subplots(figsize=(86 / 25.4, 2.50))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.30)
    ax.axvspan(13.5, 15.5, facecolor=LIGHT, zorder=0)
    available_slots = []
    for i, active in enumerate(availability):
        connected = np.flatnonzero(active) + 1
        available_slots.append(connected.tolist())
        assert np.array_equal(connected, np.arange(connected[0], connected[-1] + 1))
        color = TEAL if donors[i].any() else ORANGE
        ax.barh(
            i + 1,
            len(connected),
            left=connected[0] - 0.5,
            height=0.54,
            color=color,
            alpha=0.85,
            zorder=2,
        )
    for x in (13.5, 15.5):
        ax.axvline(x, color=GRAY, linestyle="--", linewidth=0.9, zorder=3)
    ax.text(
        14.5,
        0.35,
        r"$S=\{14,15\}$",
        ha="center",
        va="center",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )
    ax.set(
        xlim=(8.45, 20.55),
        ylim=(6.65, -0.02),
        xticks=[9, 11, 13, 15, 17, 19],
        yticks=range(1, 7),
        yticklabels=[f"EV {i}" for i in range(1, 7)],
        xlabel="Hourly slot",
    )
    fig.legend(
        handles=[
            Patch(facecolor=TEAL, label="Positive allowance"),
            Patch(facecolor=ORANGE, label=r"Must charge in $S$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.51, 0.035),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
    )
    finish(fig, "presentation_w206")
    return {
        "fleet": "W206",
        "available_slots_1based": available_slots,
        "cut_slots_1based": (slots + 1).tolist(),
        "dt_hours": dt,
        "donor_budget_kwh": budget.tolist(),
        "mandatory_charging_kwh": required.tolist(),
        "total_mandatory_charging_kwh": float(required.sum()),
        "guard_budget_kwh": guard.tolist(),
        "replenishment_budget_kwh": recharge.tolist(),
        "q": q.tolist(),
        "relaxed_cut_credit_kwh": credit.tolist(),
        "reservation_capacity_kw": retained_reservation,
        "credit_is_an_upper_bound_not_achieved_saving": True,
    }


def make_exact_class() -> dict:
    k, demand, n, q = 0.81, 4.0, 4, 1
    gamma = 1 - q / n
    replenishment = k * gamma * demand / (1 + k * gamma)
    crossover = replenishment / gamma
    assert abs(crossover - k * demand / (1 + k * gamma)) < 1e-12
    # Include the exact transition point in the sampled curve.
    total = np.unique(np.r_[np.linspace(0, 5, 251), crossover])
    power = gamma * total
    exact = np.minimum(power, replenishment)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.20, wspace=0.34)
    # The two bounds coincide with the solution on different branches.
    axes[0].axvline(crossover, color="#A6AFB4", linestyle=":", linewidth=0.9, zorder=0)
    axes[0].plot(total, exact, color=TEAL, linewidth=2.1, zorder=3)
    axes[0].plot(
        crossover,
        replenishment,
        marker="o",
        markersize=5.3,
        markerfacecolor="white",
        markeredgecolor=TEAL,
        markeredgewidth=1.5,
        zorder=4,
    )
    axes[0].text(0.98, 1.86, "Power limited", ha="center", fontsize=8)
    axes[0].text(3.52, 1.86, "Replenishment limited", ha="center", fontsize=8)
    axes[0].annotate(
        r"$\gamma B$",
        xy=(0.95, 0.95 * gamma),
        xytext=(0.44, 1.17),
        ha="center",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": "#6D777C", "lw": 0.7},
    )
    axes[0].text(3.55, 1.22, r"$k\gamma D/(1+k\gamma)$", ha="center", fontsize=8.5)
    axes[0].text(crossover + 0.14, 0.29, rf"$B^*={crossover:.3f}$ kW", fontsize=8)
    axes[0].set(
        xlim=(0, 5),
        ylim=(-0.04, 2.02),
        xticks=range(6),
        yticks=np.arange(0, 2.1, 0.5),
        xlabel=r"Total donor rating $B$ (kW)",
        ylabel="Capacity saving (kW)",
    )
    panel(axes[0], "(a)")

    ns = np.arange(1, 9)
    curves = {}
    for withdrawals, color, marker in [(0, BLUE, "o"), (1, ORANGE, "s"), (2, TEAL, "^")]:
        gamma_n = np.maximum(0, 1 - withdrawals / ns)
        values = np.minimum(gamma_n * 3, k * gamma_n * demand / (1 + k * gamma_n))
        curves[str(withdrawals)] = values.tolist()
        axes[1].plot(
            ns,
            values,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.6,
            label=rf"$q={withdrawals}$",
        )
    axes[1].set(
        xlim=(0.8, 8.2),
        ylim=(-0.04, 2.02),
        xticks=ns,
        xlabel=r"Number of donors $n$",
        ylabel="Capacity saving (kW)",
    )
    axes[1].legend(frameon=False, loc="lower right", labelspacing=0.5, handlelength=2)
    panel(axes[1], "(b)")
    for ax in axes:
        tidy(ax)
    finish(fig, "presentation_exact_class")
    return {
        "basis": "Analytical exact two-slot formula; not additional experiments",
        "D_kw": demand,
        "k": k,
        "panel_a_n": n,
        "panel_a_q": q,
        "panel_a_B_kw": total.tolist(),
        "panel_a_power_limit_kw": power.tolist(),
        "panel_a_replenishment_limit_kw": replenishment,
        "panel_a_exact_saving_kw": exact.tolist(),
        "panel_a_crossover_B_kw": crossover,
        "panel_b_B_kw": 3,
        "panel_b_n": ns.tolist(),
        "panel_b_saving_kw_by_q": curves,
    }


def make_timing(rows: list[dict], summary: dict, frontier: list[dict]) -> dict:
    assert len(rows) == 672
    by_key = {
        (row["fleet"], row["timing"], row["energy_mapping"], int(row["q"])): row for row in rows
    }
    assert len(by_key) == 672
    infeasible = [row for row in rows if row["status"] != "optimal"]
    assert len(infeasible) == 14 and {row["fleet"] for row in infeasible} == {"W303", "W308"}
    assert all(
        row["timing"] == "exact_events" and row["energy_mapping"] == "reported_as_battery_increment"
        for row in infeasible
    )
    for row in frontier:
        baseline = by_key[
            row["fleet"], "hourly_outward", "reported_as_battery_increment", int(row["q"])
        ]
        assert abs(float(baseline["M_kw"]) - float(row["M_kw"])) < 1e-7
    variant_check = []
    for record in summary["factorial_results"]:
        timing, mapping = record["timing"], record["energy_mapping"]
        all_variant = [
            row for row in rows if row["timing"] == timing and row["energy_mapping"] == mapping
        ]
        feasible_fleets = sorted(
            {row["fleet"] for row in all_variant if row["status"] == "optimal"}
        )
        assert len(feasible_fleets) == record["feasible_fleet_count"]
        counts = [
            sum(
                row["status"] == "optimal"
                and float(row["saving_vs_same_variant_reservation_kw"]) > TOL
                for row in all_variant
                if int(row["q"]) == q
            )
            for q in range(7)
        ]
        assert counts == record["positive_saving_counts_q0_to_q6"]
        variant_check.append(
            {
                "timing": timing,
                "energy_mapping": mapping,
                "feasible_fleet_count": len(feasible_fleets),
                "positive_saving_counts": counts,
                "infeasible_fleets": record["infeasible_fleets"],
            }
        )

    colors = {"reported_as_battery_increment": BLUE, "grid_consistent_increment": ORANGE}
    styles = {"hourly_outward": ("--", "o"), "exact_events": ("-", "s")}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.28, wspace=0.31)
    plotted = []
    for ax, fleet in zip(axes, ["W203", "W206"]):
        for timing in ["hourly_outward", "exact_events"]:
            for mapping in ["reported_as_battery_increment", "grid_consistent_increment"]:
                selected = [by_key[fleet, timing, mapping, q] for q in range(7)]
                assert all(row["status"] == "optimal" for row in selected)
                values = [
                    displayed_saving(row["saving_vs_same_variant_reservation_kw"])
                    for row in selected
                ]
                for q, value in enumerate(values):
                    np.testing.assert_allclose(
                        value,
                        float(selected[-1]["M_kw"]) - float(selected[q]["M_kw"]),
                        atol=TOL,
                        rtol=0,
                    )
                linestyle, marker = styles[timing]
                timing_label = "Hourly" if timing == "hourly_outward" else "Exact events"
                mapping_label = (
                    r"$d_i$" if mapping == "reported_as_battery_increment" else r"$\eta_i^c d_i$"
                )
                ax.plot(
                    range(7),
                    values,
                    color=colors[mapping],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4,
                    markerfacecolor="white" if timing == "hourly_outward" else colors[mapping],
                    linewidth=1.5,
                    label=f"{timing_label}, {mapping_label}",
                )
                plotted.append(
                    {
                        "fleet": fleet,
                        "timing": timing,
                        "energy_mapping": mapping,
                        "q": list(range(7)),
                        "capacity_saving_kw": values,
                        "same_variant_reservation_capacity_kw": float(selected[-1]["M_kw"]),
                    }
                )
        ax.set(
            xticks=range(7),
            xlim=(-0.1, 6.1),
            xlabel=r"Protected withdrawals $q$",
            ylabel="Capacity saving (kW)",
        )
        ax.set_ylim(bottom=-0.08 if fleet == "W203" else -0.25)
        tidy(ax)
        panel(ax, "(a)" if fleet == "W203" else "(b)", fleet)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.015),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.7,
        labelspacing=0.5,
    )
    finish(fig, "presentation_timing")
    return {
        "source_rows": len(rows),
        "source_feasible_rows": len(rows) - len(infeasible),
        "source_infeasible_rows": len(infeasible),
        "infeasible_display_rule": "Not plotted as zero; neither selected fleet is infeasible in any variant.",
        "all_variant_checks": variant_check,
        "plotted_series": plotted,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.bbox": None,
        }
    )
    paths = [
        CHECKS / "full_frontier_independent_lp.csv",
        CHECKS / "critical_cut_details.csv",
        CHECKS / "inputs" / "withdrawal_budget_cases" / "W206.json",
        SENSITIVITY / "timing_energy_frontiers.csv",
        SENSITIVITY / "sensitivity_summary.json",
    ]
    before = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    frontier, cuts = read_csv(paths[0]), read_csv(paths[1])
    w206 = json.loads(paths[2].read_text())
    timing, sensitivity = read_csv(paths[3]), json.loads(paths[4].read_text())
    audit = {
        "scope": "Presentation-only rebuild; frozen data and proved formulas; no optimization, sampling, or input mutation.",
        "optimizer_calls": 0,
        "canvas_width_mm": {"two_column": 178, "presentation_w206": 86},
        "font_sizes_pt": {
            "axis_labels": 9,
            "ticks": 8.5,
            "legend": 8.5,
            "dense_heatmap_labels_and_small_annotations": 8,
        },
        "saving_display_tolerance_kw": TOL,
        "saving_display_rule": "Absolute savings <= 1e-6 kW shown as zero; larger negative values fail validation.",
        "input_sha256": before,
        "presentation_contract": make_contract(),
        "presentation_frontier": make_frontier(frontier),
        "presentation_w206": make_w206(w206, cuts, frontier),
        "presentation_exact_class": make_exact_class(),
        "presentation_timing": make_timing(timing, sensitivity, frontier),
    }
    after = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    assert before == after, "Frozen sources changed during figure generation."
    audit["frozen_sources_unchanged"] = True
    audit["figure_files"] = [
        f"outputs/{stem}.{ext}"
        for stem in (
            "presentation_contract",
            "presentation_frontier",
            "presentation_w206",
            "presentation_exact_class",
            "presentation_timing",
        )
        for ext in ("pdf", "png")
    ]
    (OUT / "presentation_figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Built five vector PDF/PNG figures; validated frozen series and source hashes.")


if __name__ == "__main__":
    main()
