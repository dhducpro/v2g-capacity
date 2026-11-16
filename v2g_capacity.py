"""Capacity bounds for EV fleets with fixed charging and up to q discharge withdrawals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from replication.cplex_lp import linprog
from scipy.sparse import coo_matrix

FEASIBILITY_TOL = 1e-6
CRITICALITY_TOL = 1e-6
ENERGY_ZERO_TOL = 1e-9
POWER_ZERO_TOL = 1e-10


def _model(model):
    """Validate dimensions, broadcast scalar parameters and copy the fleet inputs."""
    a = np.asarray(model["availability"], dtype=float).copy()
    if a.ndim != 2 or not a.size:
        raise ValueError("availability must be a nonempty N by T matrix")
    n, n_slots = a.shape
    if np.any((a != 0) & (a != 1)):
        raise ValueError("Availability must be binary on every interval")
    if int(model.get("N", n)) != n or int(model.get("T", n_slots)) != n_slots:
        raise ValueError("N/T do not match availability")
    m = {"name": str(model.get("name", "unnamed")), "N": n, "T": n_slots, "availability": a}
    for key in ("duration", "L", "Ulo", "Uhi"):
        if key not in model:
            raise ValueError(f"Missing explicit model field {key}")
        m[key] = np.broadcast_to(np.asarray(model[key], float), (n_slots,)).copy()
    for key in ("E0", "Emin", "Emax", "Etarget", "Pc", "Pd", "eta_c", "eta_d"):
        m[key] = np.broadcast_to(np.asarray(model[key], float), (n,)).copy()
    if any(not np.isfinite(v).all() for v in m.values() if isinstance(v, np.ndarray)):
        raise ValueError("Model arrays must be finite")
    if np.any(m["duration"] <= 0):
        raise ValueError("Intervals must have strictly positive duration")
    if np.any(m["Pc"] < 0) or np.any(m["Pd"] < 0):
        raise ValueError("Power ratings must be nonnegative")
    for key in ("eta_c", "eta_d"):
        if np.any(m[key] <= 0) or np.any(m[key] > 1):
            raise ValueError("Efficiencies must be in (0,1]")
    if np.any(m["Ulo"] < 0) or np.any(m["Uhi"] < m["Ulo"]):
        raise ValueError("Coefficient endpoints must obey 0 <= Ulo <= Uhi")
    if (
        np.any(m["Emax"] < m["Emin"])
        or np.any(m["E0"] < m["Emin"] - 1e-10)
        or np.any(m["E0"] > m["Emax"] + 1e-10)
        or np.any(m["Etarget"] < m["Emin"] - 1e-10)
        or np.any(m["Etarget"] > m["Emax"] + 1e-10)
    ):
        raise ValueError("Initial/target energy must lie within battery bounds")
    return m


def _validate_budget(q, n):
    if isinstance(q, bool) or int(q) != q or not 0 <= q <= n:
        raise ValueError("q must be an integer from zero through N")
    return int(q)


class _ConstraintRows:
    def __init__(self):
        self.row, self.col, self.val, self.rhs = [], [], [], []

    def add(self, indices, values, rhs):
        indices = np.atleast_1d(indices).astype(int)
        values = np.broadcast_to(np.asarray(values, float), indices.shape)
        keep = values != 0
        self.row.extend([len(self.rhs)] * int(keep.sum()))
        self.col.extend(indices[keep].tolist())
        self.val.extend(values[keep].tolist())
        self.rhs.append(float(rhs))

    def matrix(self, n_variables):
        return coo_matrix(
            (self.val, (self.row, self.col)), shape=(len(self.rhs), n_variables)
        ).tocsr()


def audit_schedule(model, q, charge, discharge, M, guard=True):
    """Check physical rows directly, independently of the sparse LP matrix."""
    m = _model(model)
    q = _validate_budget(q, m["N"])
    c, d = np.asarray(charge, float), np.asarray(discharge, float)
    if c.shape != m["availability"].shape or d.shape != c.shape:
        raise ValueError("Schedule shape differs from availability")
    if not np.isfinite(c).all() or not np.isfinite(d).all() or not np.isfinite(float(M)):
        raise ValueError("Schedule powers and capacity must be finite")
    dt = m["duration"][None, :]
    energy = m["E0"][:, None] + np.cumsum(
        dt * (m["eta_c"][:, None] * c - d / m["eta_d"][:, None]), axis=1
    )
    energy_without_discharge = m["E0"][:, None] + np.cumsum(dt * m["eta_c"][:, None] * c, axis=1)
    energy_errors = [
        0.0,
        float(np.max(m["Emin"][:, None] - energy)),
        float(np.max(energy - m["Emax"][:, None])),
        float(np.max(m["Etarget"] - energy[:, -1])),
    ]
    if guard:
        energy_errors.append(float(np.max(energy_without_discharge - m["Emax"][:, None])))
    removed = np.sort(d, axis=0)[-q:].sum(axis=0) if q else np.zeros(m["T"])
    net = c.sum(axis=0) - d.sum(axis=0) + removed
    rows = np.maximum(m["L"] + m["Ulo"] * net, m["L"] + m["Uhi"] * net)
    power_error = max(
        0.0,
        -float(M),
        float(np.max(-c)),
        float(np.max(-d)),
        float(np.max(c - m["availability"] * m["Pc"][:, None])),
        float(np.max(d - m["availability"] * m["Pd"][:, None])),
        float(np.max(rows - M)),
    )
    overlap = max(0.0, float(np.max(np.minimum(c, d))))
    return {
        "energy_residual_kwh": max(energy_errors),
        "power_residual_kw": power_error,
        "overlap_kw": overlap,
        "guard_checked": bool(guard),
    }


def _recover_schedule(m, q, solution, c, d, elapsed, guard, formulation):
    if not solution.success and solution.status != 2:
        raise RuntimeError(f"Capacity solver failed: {solution.message}")
    result = {
        "status": "optimal" if solution.success else "infeasible",
        "cplex_status": solution.native_status,
        "message": str(solution.message),
        "M_kw": None,
        "energy_residual_kwh": None,
        "power_residual_kw": None,
        "overlap_kw": None,
        "q": int(q),
        "guard": bool(guard),
        "formulation": formulation,
        "elapsed_seconds": float(elapsed),
        "interval_count": m["T"],
        "minimum_subinterval_hours": float(m["duration"].min()),
        "individual_charge_deficit_kwh": float(
            np.maximum(
                0,
                m["Etarget"] - m["E0"] - m["eta_c"] * m["Pc"] * (m["availability"] @ m["duration"]),
            ).max()
        ),
    }
    if not solution.success:
        return result, None
    c, d = c.copy(), d.copy()
    k = (m["eta_c"] * m["eta_d"])[:, None]
    cycle = np.minimum(c, d / k)
    c -= cycle
    d -= k * cycle
    audit = audit_schedule(m, q, c, d, solution.fun, guard=guard)
    if (
        max(audit[k] for k in ("energy_residual_kwh", "power_residual_kw", "overlap_kw"))
        > FEASIBILITY_TOL
    ):
        raise AssertionError((m["name"], q, "Recovered schedule is infeasible", audit))
    result.update(M_kw=float(solution.fun), **audit)
    dt = m["duration"][None, :]
    schedule = {
        "charge_kw": c,
        "discharge_kw": d,
        "energy_kwh": m["E0"][:, None]
        + np.cumsum(dt * (m["eta_c"][:, None] * c - d / m["eta_d"][:, None]), axis=1),
        "no_discharge_energy_kwh": m["E0"][:, None]
        + np.cumsum(dt * m["eta_c"][:, None] * c, axis=1),
    }
    return result, schedule


def solve_capacity(model, q, guard=True):
    """Solve the robust fixed-charging contract and recover complementarity."""
    m = _model(model)
    n, n_slots = m["N"], m["T"]
    q = _validate_budget(q, n)
    n_vehicle_slots = n * n_slots
    state_start = 2 * n_vehicle_slots
    threshold_start = 3 * n_vehicle_slots
    partial_withdrawal = 0 < q < n
    excess_start = threshold_start + (n_slots if partial_withdrawal else 0)
    capacity_index = excess_start + (n_vehicle_slots if partial_withdrawal else 0)
    n_variables = capacity_index + 1
    inequalities, equalities = _ConstraintRows(), _ConstraintRows()
    for i in range(n):
        for t in range(n_slots):
            pos = i * n_slots + t
            indices = [state_start + pos, pos, n_vehicle_slots + pos]
            coefficients = [
                1.0,
                -m["duration"][t] * m["eta_c"][i],
                m["duration"][t] / m["eta_d"][i],
            ]
            rhs = m["E0"][i] if t == 0 else 0.0
            if t:
                indices.append(state_start + pos - 1)
                coefficients.append(-1.0)
            equalities.add(indices, coefficients, rhs)
        if guard:
            inequalities.add(
                np.arange(i * n_slots, (i + 1) * n_slots),
                m["duration"] * m["eta_c"][i],
                m["Emax"][i] - m["E0"][i],
            )
    for t in range(n_slots):
        charge_indices = np.arange(n) * n_slots + t
        discharge_indices = charge_indices + n_vehicle_slots
        if partial_withdrawal:
            for i in range(n):
                inequalities.add(
                    [
                        n_vehicle_slots + i * n_slots + t,
                        threshold_start + t,
                        excess_start + i * n_slots + t,
                    ],
                    [1, -1, -1],
                    0,
                )
        for u in sorted({float(m["Ulo"][t]), float(m["Uhi"][t])}):
            indices, coefficients = list(charge_indices), [u] * n
            if q < n:
                indices.extend(discharge_indices.tolist())
                coefficients.extend([-u] * n)
            if partial_withdrawal:
                indices.append(threshold_start + t)
                coefficients.append(u * q)
                indices.extend((excess_start + charge_indices).tolist())
                coefficients.extend([u] * n)
            indices.append(capacity_index)
            coefficients.append(-1.0)
            inequalities.add(indices, coefficients, -m["L"][t])
    bounds = [(0.0, float(v)) for v in (m["availability"] * m["Pc"][:, None]).ravel()]
    bounds += [(0.0, float(v)) for v in (m["availability"] * m["Pd"][:, None]).ravel()]
    for i in range(n):
        for t in range(n_slots):
            lower = max(m["Emin"][i], m["Etarget"][i]) if t == n_slots - 1 else m["Emin"][i]
            bounds.append((float(lower), float(m["Emax"][i])))
    bounds += [(0.0, None)] * (n_variables - 3 * n_vehicle_slots)
    objective = np.zeros(n_variables)
    objective[capacity_index] = 1.0
    start = perf_counter()
    solution = linprog(
        objective,
        A_ub=inequalities.matrix(n_variables),
        b_ub=inequalities.rhs,
        A_eq=equalities.matrix(n_variables),
        b_eq=equalities.rhs,
        bounds=bounds
    )
    elapsed = perf_counter() - start
    c = solution.x[:n_vehicle_slots].reshape(n, n_slots) if solution.success else None
    d = (
        solution.x[n_vehicle_slots : 2 * n_vehicle_slots].reshape(n, n_slots)
        if solution.success
        else None
    )
    return _recover_schedule(m, q, solution, c, d, elapsed, guard, "sparse_state_top_q_epigraph")


def solve_reservation(model):
    """Independent charge-only flow LP; initial surplus needs no charging."""
    m = _model(model)
    n, n_slots = m["N"], m["T"]
    n_vehicle_slots, inequalities = n * n_slots, _ConstraintRows()
    h = np.maximum(0.0, (m["Etarget"] - m["E0"]) / m["eta_c"])
    # Meeting the target exactly also satisfies the no-discharge energy guard.
    equalities = _ConstraintRows()
    for i in range(n):
        equalities.add(np.arange(i * n_slots, (i + 1) * n_slots), m["duration"], h[i])
    for t in range(n_slots):
        for u in sorted({float(m["Ulo"][t]), float(m["Uhi"][t])}):
            inequalities.add(
                np.r_[np.arange(n) * n_slots + t, n_vehicle_slots],
                np.r_[np.repeat(u, n), -1.0],
                -m["L"][t],
            )
    bounds = [(0.0, float(v)) for v in (m["availability"] * m["Pc"][:, None]).ravel()]
    bounds.append((0.0, None))
    objective = np.zeros(n_vehicle_slots + 1)
    objective[-1] = 1.0
    start = perf_counter()
    solution = linprog(
        objective,
        A_ub=inequalities.matrix(n_vehicle_slots + 1),
        b_ub=inequalities.rhs,
        A_eq=equalities.matrix(n_vehicle_slots + 1),
        b_eq=equalities.rhs,
        bounds=bounds
    )
    elapsed = perf_counter() - start
    c = solution.x[:n_vehicle_slots].reshape(n, n_slots) if solution.success else None
    d = np.zeros((n, n_slots)) if solution.success else None
    return _recover_schedule(m, n, solution, c, d, elapsed, True, "charge_only_flow_lp")


@dataclass
class _Edge:
    to: int
    reverse: int
    capacity: float


class _Dinic:
    """Deterministic real-capacity flow, used only for cut candidate extraction."""

    def __init__(self, vertices, epsilon=1e-12):
        self.graph = [[] for _ in range(vertices)]
        self.epsilon = epsilon

    def add(self, a, b, cap):
        forward = _Edge(b, len(self.graph[b]), float(cap))
        backward = _Edge(a, len(self.graph[a]), 0.0)
        self.graph[a].append(forward)
        self.graph[b].append(backward)

    def flow(self, source, sink):
        value = 0.0
        while True:
            level = [-1] * len(self.graph)
            level[source] = 0
            queue = deque([source])
            while queue:
                v = queue.popleft()
                for e in self.graph[v]:
                    if e.capacity > self.epsilon and level[e.to] < 0:
                        level[e.to] = level[v] + 1
                        queue.append(e.to)
            if level[sink] < 0:
                break
            current = [0] * len(self.graph)

            def push(v, amount):
                if v == sink:
                    return amount
                while current[v] < len(self.graph[v]):
                    e = self.graph[v][current[v]]
                    if e.capacity > self.epsilon and level[e.to] == level[v] + 1:
                        sent = push(e.to, min(amount, e.capacity))
                        if sent > self.epsilon:
                            e.capacity -= sent
                            self.graph[e.to][e.reverse].capacity += sent
                            return sent
                    current[v] += 1
                return 0.0

            while True:
                sent = push(source, float("inf"))
                if sent <= self.epsilon:
                    break
                value += sent
        return value

    def source_reachable(self, source):
        seen, queue = {source}, deque([source])
        while queue:
            v = queue.popleft()
            for e in self.graph[v]:
                if e.capacity > self.epsilon and e.to not in seen:
                    seen.add(e.to)
                    queue.append(e.to)
        return seen

    def can_reach_sink(self, sink):
        # The complement is the maximal source side of a minimum cut.
        seen, queue = {sink}, deque([sink])
        while queue:
            v = queue.popleft()
            for e in self.graph[v]:
                reverse_edge = self.graph[e.to][e.reverse]
                if reverse_edge.capacity > self.epsilon and e.to not in seen:
                    seen.add(e.to)
                    queue.append(e.to)
        return seen


def _flow_cuts(m, capacity):
    n, n_slots = m["N"], m["T"]
    h = np.maximum(0.0, (m["Etarget"] - m["E0"]) / m["eta_c"])
    pc_energy = m["availability"] * m["Pc"][:, None] * m["duration"][None, :]
    # Scale the flow network to keep residual tolerances independent of its energy units.
    scale = max(1.0, float(h.sum()))
    source, sink = n + n_slots, n + n_slots + 1
    flow = _Dinic(n + n_slots + 2)
    for i in range(n):
        flow.add(source, i, h[i] / scale)
        for t in np.flatnonzero(pc_energy[i] > 0):
            flow.add(i, n + int(t), pc_energy[i, t] / scale)
    for t in range(n_slots):
        cap = (
            (capacity - m["L"][t]) * m["duration"][t] / m["Uhi"][t]
            if m["Uhi"][t] > 0
            else float(pc_energy[:, t].sum())
        )
        flow.add(n + t, sink, max(0.0, cap) / scale)
    achieved = flow.flow(source, sink) * scale
    low = flow.source_reachable(source)
    high = set(range(n + n_slots + 2)) - flow.can_reach_sink(sink)
    return achieved, [[t for t in range(n_slots) if n + t in side] for side in (low, high)]


def _cut_data(m, slots, reservation_kw):
    s = np.asarray(slots, dtype=int)
    if not len(s) or np.any(m["Uhi"][s] <= 0):
        return None
    h = (m["Etarget"] - m["E0"]) / m["eta_c"]
    charging = m["availability"] * m["Pc"][:, None] * m["duration"][None, :]
    outside_mask = np.ones(m["T"], dtype=bool)
    outside_mask[s] = False
    # Summing outside slots directly avoids cancellation when the outside set is empty.
    outside = charging[:, outside_mask].sum(axis=1)
    phi = float(np.maximum(h - outside, 0.0).sum())
    weight = float((m["duration"][s] / m["Uhi"][s]).sum())
    ratio = (phi + np.sum(m["duration"][s] * m["L"][s] / m["Uhi"][s])) / weight
    energy = np.minimum(
        m["eta_d"] * (m["Emax"] - m["Etarget"]),
        m["eta_c"] * m["eta_d"] * np.maximum(outside - h, 0.0),
    )
    energy = np.maximum(energy, 0.0)
    pd = m["availability"][:, s] * m["Pd"][:, None]
    # A tiny energy allowance can supply substantial power over a short interval.
    counts = ((energy[:, None] > 0.0) & (pd > 0.0)).sum(axis=0)
    diagnostic_counts = ((energy[:, None] > ENERGY_ZERO_TOL) & (pd > POWER_ZERO_TOL)).sum(axis=0)
    return {
        "slots": s.tolist(),
        "slots_1based": (s + 1).tolist(),
        "h_grid_kwh": h.tolist(),
        "outside_charging_energy_kwh": outside.tolist(),
        "forced_charging_energy_kwh": phi,
        "weight_hours": weight,
        "implied_capacity_kw": float(ratio),
        "M_reservation_kw": float(reservation_kw),
        "criticality_error_kw": abs(float(ratio) - float(reservation_kw)),
        "criticality_slack_kw": max(0.0, float(reservation_kw) - float(ratio)),
        "donor_budget_kwh": energy.tolist(),
        "donor_count_by_slot": counts.tolist(),
        "q_certificate": int(counts.max()),
        "q_tolerance_count_diagnostic_only": int(diagnostic_counts.max()),
        "energy_zero_tolerance_kwh": ENERGY_ZERO_TOL,
        "power_zero_tolerance_kw": POWER_ZERO_TOL,
    }


def reservation_cut(model, reservation_kw=None):
    """Find candidate minimum cuts and verify their ratios at the reservation optimum."""
    m = _model(model)
    if reservation_kw is None:
        reservation, _ = solve_reservation(m)
        if reservation["status"] != "optimal":
            return {
                "status": "infeasible_reservation",
                "M_reservation_kw": None,
                "cuts": [],
                "selected_cut": None,
                "reservation_solver": reservation,
            }
        reservation_kw = reservation["M_kw"]
    reserve = float(reservation_kw)
    if not np.isfinite(reserve):
        raise ValueError("Reservation capacity must be finite")
    floor = max(0.0, float(m["L"].max()))
    record = {
        "status": None,
        "M_reservation_kw": reserve,
        "background_floor_kw": floor,
        "cuts": [],
        "selected_cut": None,
        "criticality_tolerance_kw": CRITICALITY_TOL,
        "method": "Verified real-capacity min-cut candidates below charge-only LP optimum",
        "all_critical_cuts_enumerated": False,
        "uses_protected_capacity_outcomes": False,
        "candidate_attempts": [],
    }
    if reserve <= floor + CRITICALITY_TOL:
        record["status"] = "background_dominated_no_certificate"
        return record
    first = min((reserve - floor) / 4.0, max(1e-5, abs(reserve) * 1e-5))
    candidates = {}
    for factor in (1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001):
        epsilon = first * factor
        achieved, sets = _flow_cuts(m, reserve - epsilon)
        accepted = 0
        for slots in sets:
            cut = _cut_data(m, slots, reserve)
            if cut is not None and cut["criticality_error_kw"] <= CRITICALITY_TOL:
                key = tuple(cut["slots"])
                if key not in candidates:
                    cut["extraction_capacity_offset_kw"] = epsilon
                    candidates[key] = cut
                    accepted += 1
        record["candidate_attempts"].append(
            {
                "offset_kw": epsilon,
                "flow_energy_kwh": float(achieved),
                "new_verified_candidates": accepted,
            }
        )
    cuts = sorted(
        candidates.values(), key=lambda x: (x["q_certificate"], len(x["slots"]), x["slots"])
    )
    record.update(
        status="verified_critical_cut" if cuts else "unresolved_critical_cut",
        cuts=cuts,
        selected_cut=cuts[0] if cuts else None,
    )
    return record


def _dual_credit_bound(row_duals, energy, power, dt, q):
    """Bound donor credit with a feasible withdrawal allocation and fractional knapsack."""
    n, n_slots = power.shape
    weights = np.empty((n, n_slots))
    for i in range(n):
        first = i * (n_slots + 1) + 1
        weights[i] = -np.asarray(row_duals[first : first + n_slots]) / dt
    weights = np.clip(weights, 0.0, 1.0)
    column_sum = weights.sum(axis=0)
    # Leave room for rounding when enforcing the withdrawal budget in each slot.
    safe_q = q * (1.0 - 64 * np.finfo(float).eps)
    over = column_sum > safe_q
    weights[:, over] *= (safe_q / column_sum[over])[None, :]
    credit_upper = 0.0
    for i in range(n):
        remaining = float(energy[i])
        coefficients = 1.0 - weights[i]
        for t in np.argsort(-coefficients, kind="stable"):
            take = min(remaining, float(dt[t] * power[i, t]))
            credit_upper += float(coefficients[t]) * take
            remaining = max(0.0, remaining - take)
            if remaining == 0.0:
                break
    return credit_upper


def compute_certificate(model, cut, q):
    """Bound capacity savings using the donor energy available on a verified cut."""
    m = _model(model)
    q = _validate_budget(q, m["N"])
    if cut is None or cut.get("criticality_error_kw", float("inf")) > CRITICALITY_TOL:
        raise ValueError("A verified nonempty critical cut is required")
    fresh = _cut_data(m, cut["slots"], cut["M_reservation_kw"])
    if fresh is None or fresh["criticality_error_kw"] > CRITICALITY_TOL:
        raise ValueError("The critical cut does not match these model inputs")
    s = np.asarray(fresh["slots"], int)
    energy = np.asarray(fresh["donor_budget_kwh"], float)
    power = m["availability"][:, s] * m["Pd"][:, None]
    dt = m["duration"][s]
    n, n_slots = power.shape
    start = perf_counter()
    if q >= n or q >= fresh["q_certificate"]:
        credit = 0.0
        credit_upper = 0.0
        method = "donor_count_zero_certificate"
    elif q == 0:
        credit = float(np.minimum(power @ dt, energy).sum())
        credit_upper = credit
        method = "separable_no_withdrawal_energy_formula"
    else:
        n_vehicle_slots = n * n_slots
        threshold_start, excess_start, n_variables = (
            n_vehicle_slots,
            n_vehicle_slots + n_slots,
            2 * n_vehicle_slots + n_slots,
        )
        objective = np.r_[-np.tile(dt, n), q * dt, np.tile(dt, n)]
        inequalities = _ConstraintRows()
        for i in range(n):
            inequalities.add(np.arange(i * n_slots, (i + 1) * n_slots), dt, energy[i])
            for t in range(n_slots):
                inequalities.add(
                    [i * n_slots + t, threshold_start + t, excess_start + i * n_slots + t],
                    [1, -1, -1],
                    0.0,
                )
        bounds = [(0.0, float(v)) for v in power.ravel()] + [(0.0, None)] * (
            n_variables - n_vehicle_slots
        )
        solution = linprog(
            objective,
            A_ub=inequalities.matrix(n_variables),
            b_ub=inequalities.rhs,
            bounds=bounds
        )
        if not solution.success:
            raise RuntimeError(f"Credit LP failed: {solution.message}")
        credit = max(0.0, -float(solution.fun))
        z = solution.x[:n_vehicle_slots].reshape(n, n_slots)
        actual = float(np.dot(dt, z.sum(axis=0) - np.sort(z, axis=0)[-q:].sum(axis=0)))
        if (
            max(
                0.0,
                float(np.max(z @ dt - energy)),
                float(np.max(z - power)),
                float(np.max(-z)),
                abs(actual - credit),
            )
            > FEASIBILITY_TOL
        ):
            raise AssertionError("Direct weighted credit-LP audit failed")
        dual_bound = _dual_credit_bound(solution.ineqlin.marginals, energy, power, dt, q)
        credit_upper = max(credit, dual_bound)
        method = "weighted_top_q_credit_lp"
    raw_bound = credit / fresh["weight_hours"]
    upper_bound = credit_upper / fresh["weight_hours"] + fresh["criticality_slack_kw"]
    return {
        "q": q,
        "R_kwh": credit_upper,
        "R_primal_kwh": credit,
        "credit_upper_gap_kwh": max(0.0, credit_upper - credit),
        "raw_bound_kw": raw_bound,
        "bound_kw": upper_bound,
        "q_certificate": fresh["q_certificate"],
        "count_condition_satisfied": bool(q >= fresh["q_certificate"]),
        "certifies_zero_saving": bool(upper_bound <= FEASIBILITY_TOL),
        "zero_saving_tolerance_kw": FEASIBILITY_TOL,
        "slots": fresh["slots"],
        "criticality_error_kw": fresh["criticality_error_kw"],
        "criticality_slack_kw": fresh["criticality_slack_kw"],
        "method": method,
        "elapsed_seconds": perf_counter() - start,
        "units": {"R": "kWh", "weight": "h", "bound": "kW"},
    }
