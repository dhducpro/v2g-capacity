"""Capacity and critical-cut computations on unequal-duration intervals.

This module implements the declared fixed-charging/discharge-withdrawal
contract. It does not select study inputs and does not optimize which critical
cut is best over all possible critical cuts. Cut candidates depend only on the
charging reservation problem and input arrays, never protected-capacity
outcomes. Every retained candidate is checked against the reservation value.

Public API
----------
solve_capacity(model, q, guard=True) -> (result, schedule or None)
solve_reservation(model) -> (result, charge-only schedule or None)
reservation_cut(model, reservation_kw=None) -> extraction record
compute_certificate(model, cut, q) -> weighted-credit record
audit_schedule(model, q, charge, discharge, M, guard=True) -> residual record

The model fields are name, availability (N,T), duration (T, hours), L (T,kW),
E0/Emin/Emax/Etarget (N,kWh), Pc/Pd (N,kW), eta_c/eta_d (N), Ulo/Uhi
(T, dimensionless; scalar broadcasting is accepted). N/T are optional and,
when supplied, are checked. All cut energy quantities are GRID-SIDE kWh;
the cut weight has units hours and the resulting saving bound has units kW.

The capacity solver uses post-interval battery states and a top-q epigraph,
with O(NT) variables, rows and nonzero entries. The guard needs only its final
row because charging is nonnegative; the direct audit checks every prefix.
Small-instance exhaustive comparisons live in check_weighted_capacity.py.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cplex_lp import linprog
from scipy.sparse import coo_matrix


FEASIBILITY_TOL = 1e-6
CRITICALITY_TOL = 1e-6
ENERGY_ZERO_TOL = 1e-9
POWER_ZERO_TOL = 1e-10


def _model(model):
    """Validate and copy inputs; never mutate a supplied case."""
    a = np.asarray(model["availability"], dtype=float).copy()
    if a.ndim != 2 or not a.size:
        raise ValueError("availability must be a nonempty N by T matrix")
    n, nt = a.shape
    if np.any((a != 0) & (a != 1)):
        raise ValueError("Availability must be binary on every interval")
    if int(model.get("N", n)) != n or int(model.get("T", nt)) != nt:
        raise ValueError("N/T do not match availability")
    m = {"name": str(model.get("name", "unnamed")), "N": n, "T": nt,
         "availability": a}
    for key in ("duration", "L", "Ulo", "Uhi"):
        if key not in model:
            raise ValueError(f"Missing explicit model field {key}")
        m[key] = np.broadcast_to(np.asarray(model[key], float), (nt,)).copy()
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
    if (np.any(m["Emax"] < m["Emin"]) or np.any(m["E0"] < m["Emin"] - 1e-10)
            or np.any(m["E0"] > m["Emax"] + 1e-10)
            or np.any(m["Etarget"] < m["Emin"] - 1e-10)
            or np.any(m["Etarget"] > m["Emax"] + 1e-10)):
        raise ValueError("Initial/target energy must lie within battery bounds")
    return m


def _q(q, n):
    if isinstance(q, bool) or int(q) != q or not 0 <= q <= n:
        raise ValueError("q must be an integer from zero through N")
    return int(q)


class _Rows:
    def __init__(self):
        self.row, self.col, self.val, self.rhs = [], [], [], []

    def add(self, ix, values, rhs):
        indices = np.atleast_1d(ix).astype(int)
        values = np.broadcast_to(np.asarray(values, float), indices.shape)
        keep = values != 0
        self.row.extend([len(self.rhs)] * int(keep.sum()))
        self.col.extend(indices[keep].tolist())
        self.val.extend(values[keep].tolist())
        self.rhs.append(float(rhs))

    def matrix(self, nv):
        return coo_matrix((self.val, (self.row, self.col)),
                          shape=(len(self.rhs), nv)).tocsr()


def audit_schedule(model, q, charge, discharge, M, guard=True):
    """Check physical rows directly, independently of the sparse LP matrix."""
    m = _model(model)
    q = _q(q, m["N"])
    c, d = np.asarray(charge, float), np.asarray(discharge, float)
    if c.shape != m["availability"].shape or d.shape != c.shape:
        raise ValueError("Schedule shape differs from availability")
    if not np.isfinite(c).all() or not np.isfinite(d).all() or not np.isfinite(float(M)):
        raise ValueError("Schedule powers and capacity must be finite")
    dt = m["duration"][None, :]
    energy = m["E0"][:, None] + np.cumsum(
        dt * (m["eta_c"][:, None] * c - d / m["eta_d"][:, None]), axis=1)
    nd = m["E0"][:, None] + np.cumsum(dt * m["eta_c"][:, None] * c, axis=1)
    energy_errors = [0., float(np.max(m["Emin"][:, None] - energy)),
                     float(np.max(energy - m["Emax"][:, None])),
                     float(np.max(m["Etarget"] - energy[:, -1]))]
    if guard:
        energy_errors.append(float(np.max(nd - m["Emax"][:, None])))
    removed = np.sort(d, axis=0)[-q:].sum(axis=0) if q else np.zeros(m["T"])
    net = c.sum(axis=0) - d.sum(axis=0) + removed
    rows = np.maximum(m["L"] + m["Ulo"] * net, m["L"] + m["Uhi"] * net)
    power_error = max(0., -float(M), float(np.max(-c)), float(np.max(-d)),
                      float(np.max(c - m["availability"] * m["Pc"][:, None])),
                      float(np.max(d - m["availability"] * m["Pd"][:, None])),
                      float(np.max(rows - M)))
    overlap = max(0., float(np.max(np.minimum(c, d))))
    return {"energy_residual_kwh": max(energy_errors),
            "power_residual_kw": power_error, "overlap_kw": overlap,
            "guard_checked": bool(guard)}


def _finish(m, q, ans, c, d, elapsed, guard, formulation):
    result = {"status": "optimal" if ans.success else "infeasible" if ans.status == 2 else "solver_error",
              "cplex_status": ans.native_status, "message": str(ans.message),
              "M_kw": None, "energy_residual_kwh": None,
              "power_residual_kw": None, "overlap_kw": None,
              "q": int(q), "guard": bool(guard), "formulation": formulation,
              "elapsed_seconds": float(elapsed), "interval_count": m["T"],
              "minimum_subinterval_hours": float(m["duration"].min()),
              "individual_charge_deficit_kwh": float(np.maximum(0,
                  m["Etarget"] - m["E0"] - m["eta_c"] * m["Pc"] *
                  (m["availability"] @ m["duration"])).max())}
    if not ans.success:
        if ans.status != 2:
            raise RuntimeError(f"Capacity solver failed: {ans.message}")
        return result, None
    c, d = c.copy(), d.copy()
    k = (m["eta_c"] * m["eta_d"])[:, None]
    cycle = np.minimum(c, d / k)
    c -= cycle
    d -= k * cycle
    audit = audit_schedule(m, q, c, d, ans.fun, guard=guard)
    if max(audit[k] for k in ("energy_residual_kwh", "power_residual_kw", "overlap_kw")) > FEASIBILITY_TOL:
        raise AssertionError((m["name"], q, "Recovered schedule is infeasible", audit))
    result.update(M_kw=float(ans.fun), **audit)
    dt = m["duration"][None, :]
    schedule = {"charge_kw": c, "discharge_kw": d,
                "energy_kwh": m["E0"][:, None] + np.cumsum(
                    dt * (m["eta_c"][:, None] * c - d / m["eta_d"][:, None]), axis=1),
                "no_discharge_energy_kwh": m["E0"][:, None] + np.cumsum(
                    dt * m["eta_c"][:, None] * c, axis=1)}
    return result, schedule


def solve_capacity(model, q, guard=True):
    """Solve the robust fixed-charging contract and recover complementarity."""
    m = _model(model)
    n, nt = m["N"], m["T"]
    q = _q(q, n)
    nz = n * nt
    e0 = 2 * nz
    theta0 = 3 * nz
    middle = 0 < q < n
    r0 = theta0 + (nt if middle else 0)
    Midx = r0 + (nz if middle else 0)
    nv = Midx + 1
    ub, eq = _Rows(), _Rows()
    for i in range(n):
        for t in range(nt):
            pos = i * nt + t
            ix = [e0 + pos, pos, nz + pos]
            vals = [1., -m["duration"][t] * m["eta_c"][i],
                    m["duration"][t] / m["eta_d"][i]]
            rhs = m["E0"][i] if t == 0 else 0.
            if t:
                ix.append(e0 + pos - 1); vals.append(-1.)
            eq.add(ix, vals, rhs)
        if guard:
            ub.add(np.arange(i * nt, (i + 1) * nt),
                   m["duration"] * m["eta_c"][i], m["Emax"][i] - m["E0"][i])
    for t in range(nt):
        cix = np.arange(n) * nt + t
        dix = cix + nz
        if middle:
            for i in range(n):
                ub.add([nz + i * nt + t, theta0 + t, r0 + i * nt + t], [1, -1, -1], 0)
        for u in sorted({float(m["Ulo"][t]), float(m["Uhi"][t])}):
            ix, vals = list(cix), [u] * n
            if q < n:
                ix.extend(dix.tolist()); vals.extend([-u] * n)
            if middle:
                ix.append(theta0 + t); vals.append(u * q)
                ix.extend((r0 + cix).tolist()); vals.extend([u] * n)
            ix.append(Midx); vals.append(-1.)
            ub.add(ix, vals, -m["L"][t])
    bounds = [(0., float(v)) for v in (m["availability"] * m["Pc"][:, None]).ravel()]
    bounds += [(0., float(v)) for v in (m["availability"] * m["Pd"][:, None]).ravel()]
    for i in range(n):
        for t in range(nt):
            lower = max(m["Emin"][i], m["Etarget"][i]) if t == nt - 1 else m["Emin"][i]
            bounds.append((float(lower), float(m["Emax"][i])))
    bounds += [(0., None)] * (nv - 3 * nz)
    objective = np.zeros(nv); objective[Midx] = 1.
    start = perf_counter()
    ans = linprog(objective, A_ub=ub.matrix(nv), b_ub=ub.rhs,
                  A_eq=eq.matrix(nv), b_eq=eq.rhs, bounds=bounds)
    elapsed = perf_counter() - start
    c = ans.x[:nz].reshape(n, nt) if ans.success else None
    d = ans.x[nz:2*nz].reshape(n, nt) if ans.success else None
    return _finish(m, q, ans, c, d, elapsed, guard, "sparse_state_top_q_epigraph")


def solve_reservation(model):
    """Independent charge-only flow LP; initial surplus needs no charging."""
    m = _model(model)
    n, nt = m["N"], m["T"]
    nz, ub = n * nt, _Rows()
    h = np.maximum(0., (m["Etarget"] - m["E0"]) / m["eta_c"])
    # Equality avoids arbitrary excess energy and automatically satisfies the
    # initial/target-interpolating battery trajectory and common guard.
    eq = _Rows()
    for i in range(n):
        eq.add(np.arange(i * nt, (i + 1) * nt), m["duration"], h[i])
    for t in range(nt):
        for u in sorted({float(m["Ulo"][t]), float(m["Uhi"][t])}):
            ub.add(np.r_[np.arange(n) * nt + t, nz], np.r_[np.repeat(u, n), -1.], -m["L"][t])
    bounds = [(0., float(v)) for v in (m["availability"] * m["Pc"][:, None]).ravel()]
    bounds.append((0., None))
    objective = np.zeros(nz + 1); objective[-1] = 1.
    start = perf_counter()
    ans = linprog(objective, A_ub=ub.matrix(nz + 1), b_ub=ub.rhs,
                  A_eq=eq.matrix(nz + 1), b_eq=eq.rhs, bounds=bounds)
    elapsed = perf_counter() - start
    c = ans.x[:nz].reshape(n, nt) if ans.success else None
    d = np.zeros((n, nt)) if ans.success else None
    return _finish(m, n, ans, c, d, elapsed, True, "charge_only_flow_lp")


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
        backward = _Edge(a, len(self.graph[a]), 0.)
        self.graph[a].append(forward); self.graph[b].append(backward)

    def flow(self, source, sink):
        value = 0.
        while True:
            level = [-1] * len(self.graph); level[source] = 0
            queue = deque([source])
            while queue:
                v = queue.popleft()
                for e in self.graph[v]:
                    if e.capacity > self.epsilon and level[e.to] < 0:
                        level[e.to] = level[v] + 1; queue.append(e.to)
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
                return 0.

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
                    seen.add(e.to); queue.append(e.to)
        return seen

    def can_reach_sink(self, sink):
        # Reverse traversal of positive residual arcs gives the maximal
        # source-side minimum cut as the complement of this set.
        seen, queue = {sink}, deque([sink])
        while queue:
            v = queue.popleft()
            for e in self.graph[v]:
                reverse_edge = self.graph[e.to][e.reverse]
                if reverse_edge.capacity > self.epsilon and e.to not in seen:
                    seen.add(e.to); queue.append(e.to)
        return seen


def _flow_cuts(m, capacity):
    n, nt = m["N"], m["T"]
    h = np.maximum(0., (m["Etarget"] - m["E0"]) / m["eta_c"])
    pc_energy = m["availability"] * m["Pc"][:, None] * m["duration"][None, :]
    # Normalize capacities to keep residual comparisons away from arbitrary
    # kWh scales. No integer quantization or rounding of energy is used.
    scale = max(1., float(h.sum()))
    source, sink = n + nt, n + nt + 1
    flow = _Dinic(n + nt + 2)
    for i in range(n):
        flow.add(source, i, h[i] / scale)
        for t in np.flatnonzero(pc_energy[i] > 0):
            flow.add(i, n + int(t), pc_energy[i, t] / scale)
    for t in range(nt):
        cap = ((capacity - m["L"][t]) * m["duration"][t] / m["Uhi"][t]
               if m["Uhi"][t] > 0 else float(pc_energy[:, t].sum()))
        flow.add(n + t, sink, max(0., cap) / scale)
    achieved = flow.flow(source, sink) * scale
    low = flow.source_reachable(source)
    high = set(range(n + nt + 2)) - flow.can_reach_sink(sink)
    return achieved, [[t for t in range(nt) if n + t in side] for side in (low, high)]


def _cut_data(m, slots, reservation_kw):
    s = np.asarray(slots, dtype=int)
    if not len(s) or np.any(m["Uhi"][s] <= 0):
        return None
    h = (m["Etarget"] - m["E0"]) / m["eta_c"]
    charging = m["availability"] * m["Pc"][:, None] * m["duration"][None, :]
    outside_mask = np.ones(m["T"], dtype=bool)
    outside_mask[s] = False
    # Sum the actual outside intervals rather than subtracting two large
    # totals; a genuinely empty outside set then has exactly zero energy.
    outside = charging[:, outside_mask].sum(axis=1)
    phi = float(np.maximum(h - outside, 0.).sum())
    weight = float((m["duration"][s] / m["Uhi"][s]).sum())
    ratio = (phi + np.sum(m["duration"][s] * m["L"][s] / m["Uhi"][s])) / weight
    energy = np.minimum(m["eta_d"] * (m["Emax"] - m["Etarget"]),
                        m["eta_c"] * m["eta_d"] * np.maximum(outside - h, 0.))
    energy = np.maximum(energy, 0.)
    pd = m["availability"][:, s] * m["Pd"][:, None]
    # Mathematical zero uses strict positivity. An energy-scale tolerance
    # must not erase a donor whose tiny kWh allowance produces appreciable
    # kW credit on a very short critical interval.
    counts = ((energy[:, None] > 0.) & (pd > 0.)).sum(axis=0)
    diagnostic_counts = ((energy[:, None] > ENERGY_ZERO_TOL) & (pd > POWER_ZERO_TOL)).sum(axis=0)
    return {"slots": s.tolist(), "slots_1based": (s + 1).tolist(),
            "h_grid_kwh": h.tolist(), "outside_charging_energy_kwh": outside.tolist(),
            "forced_charging_energy_kwh": phi, "weight_hours": weight,
            "implied_capacity_kw": float(ratio), "M_reservation_kw": float(reservation_kw),
            "criticality_error_kw": abs(float(ratio) - float(reservation_kw)),
            "criticality_slack_kw": max(0., float(reservation_kw) - float(ratio)),
            "donor_budget_kwh": energy.tolist(), "donor_count_by_slot": counts.tolist(),
            "q_certificate": int(counts.max()),
            "q_tolerance_count_diagnostic_only": int(diagnostic_counts.max()),
            "energy_zero_tolerance_kwh": ENERGY_ZERO_TOL,
            "power_zero_tolerance_kw": POWER_ZERO_TOL}


def reservation_cut(model, reservation_kw=None):
    """Extract verified critical time cuts without enumerating EV subsets.

    A deterministic sequence of capacities just below the charge-only LP
    optimum generates source-reachable and maximal-source-side min-cuts.
    The ratio of each candidate is verified at the ORIGINAL optimum. An
    approximate deficient cut is never accepted solely because a min-cut
    routine returned it. Multiple verified candidates are retained; the
    selected one minimizes the input-only donor threshold, then cardinality,
    then lexicographic interval order among those detected candidates.
    This is not exhaustive minimization over all critical cuts.
    """
    m = _model(model)
    if reservation_kw is None:
        reservation, _ = solve_reservation(m)
        if reservation["status"] != "optimal":
            return {"status": "infeasible_reservation", "M_reservation_kw": None,
                    "cuts": [], "selected_cut": None, "reservation_solver": reservation}
        reservation_kw = reservation["M_kw"]
    reserve = float(reservation_kw)
    if not np.isfinite(reserve):
        raise ValueError("Reservation capacity must be finite")
    floor = max(0., float(m["L"].max()))
    record = {"status": None, "M_reservation_kw": reserve,
              "background_floor_kw": floor, "cuts": [], "selected_cut": None,
              "criticality_tolerance_kw": CRITICALITY_TOL,
              "method": "Verified real-capacity min-cut candidates below charge-only LP optimum",
              "all_critical_cuts_enumerated": False,
              "uses_protected_capacity_outcomes": False, "candidate_attempts": []}
    if reserve <= floor + CRITICALITY_TOL:
        record["status"] = "background_dominated_no_certificate"
        return record
    first = min((reserve - floor) / 4., max(1e-5, abs(reserve) * 1e-5))
    candidates = {}
    for factor in (1., .1, .01, .001, .0001, .00001):
        epsilon = first * factor
        achieved, sets = _flow_cuts(m, reserve - epsilon)
        accepted = 0
        for slots in sets:
            cut = _cut_data(m, slots, reserve)
            if cut is not None and cut["criticality_error_kw"] <= CRITICALITY_TOL:
                key = tuple(cut["slots"])
                if key not in candidates:
                    cut["extraction_capacity_offset_kw"] = epsilon
                    candidates[key] = cut; accepted += 1
        record["candidate_attempts"].append({"offset_kw": epsilon,
            "flow_energy_kwh": float(achieved), "new_verified_candidates": accepted})
    cuts = sorted(candidates.values(), key=lambda x: (x["q_certificate"], len(x["slots"]), x["slots"]))
    record.update(status="verified_critical_cut" if cuts else "unresolved_critical_cut",
                  cuts=cuts, selected_cut=cuts[0] if cuts else None)
    return record


def compute_certificate(model, cut, q):
    """Compute the weighted relaxed credit for one VERIFIED critical cut."""
    m = _model(model)
    q = _q(q, m["N"])
    if cut is None or cut.get("criticality_error_kw", float("inf")) > CRITICALITY_TOL:
        raise ValueError("A verified nonempty critical cut is required")
    fresh = _cut_data(m, cut["slots"], cut["M_reservation_kw"])
    if fresh is None or fresh["criticality_error_kw"] > CRITICALITY_TOL:
        raise ValueError("The critical cut does not match these model inputs")
    s = np.asarray(fresh["slots"], int)
    energy = np.asarray(fresh["donor_budget_kwh"], float)
    power = m["availability"][:, s] * m["Pd"][:, None]
    dt = m["duration"][s]
    n, nt = power.shape
    start = perf_counter()
    if q >= n or q >= fresh["q_certificate"]:
        # The zero test is itself the analytical result; do not solve a
        # degenerate credit LP merely to reproduce that test numerically.
        credit = 0.
        credit_upper = 0.
        method = "donor_count_zero_certificate"
    elif q == 0:
        credit = float(np.minimum(power @ dt, energy).sum())
        credit_upper = credit
        method = "separable_no_withdrawal_energy_formula"
    else:
        nz = n * nt
        theta0, r0, nv = nz, nz + nt, 2 * nz + nt
        objective = np.r_[-np.tile(dt, n), q * dt, np.tile(dt, n)]
        ub = _Rows()
        for i in range(n):
            ub.add(np.arange(i * nt, (i + 1) * nt), dt, energy[i])
            for t in range(nt):
                ub.add([i * nt + t, theta0 + t, r0 + i * nt + t], [1, -1, -1], 0.)
        bounds = [(0., float(v)) for v in power.ravel()] + [(0., None)] * (nv - nz)
        ans = linprog(objective, A_ub=ub.matrix(nv), b_ub=ub.rhs,
                      bounds=bounds)
        if not ans.success:
            raise RuntimeError(f"Credit LP failed: {ans.message}")
        credit = max(0., -float(ans.fun))
        z = ans.x[:nz].reshape(n, nt)
        actual = float(np.dot(dt, z.sum(axis=0) - np.sort(z, axis=0)[-q:].sum(axis=0)))
        if max(0., float(np.max(z @ dt - energy)), float(np.max(z - power)),
               float(np.max(-z)), abs(actual - credit)) > FEASIBILITY_TOL:
            raise AssertionError("Direct weighted credit-LP audit failed")
        # A primal value alone is not an upper bound on a maximization LP.
        # Construct a feasible withdrawal allocation from the row duals.
        # For any 0<=w_it<=1 with sum_i w_it<=q, v_q(z_t) <= sum_i(1-w_it)z_it.
        # Its donor-wise continuous-knapsack maxima give a conservative
        # upper bound, even if solver tolerances round a small primal R to 0.
        weights = np.empty((n, nt))
        for i in range(n):
            first = i * (nt + 1) + 1
            weights[i] = -np.asarray(ans.ineqlin.marginals[first:first+nt]) / dt
        weights = np.clip(weights, 0., 1.)
        column_sum = weights.sum(axis=0)
        # Leave a roundoff-sized interior margin so summation cannot move a
        # normalized withdrawal column above q. Reducing weights only makes
        # the upper bound more conservative.
        safe_q = q * (1. - 64 * np.finfo(float).eps)
        over = column_sum > safe_q
        weights[:, over] *= (safe_q / column_sum[over])[None, :]
        credit_upper = 0.
        for i in range(n):
            remaining = float(energy[i])
            coefficients = 1. - weights[i]
            # Stable sorting makes ties deterministic; any tie order gives
            # the same exact fractional-knapsack value.
            for t in np.argsort(-coefficients, kind="stable"):
                take = min(remaining, float(dt[t] * power[i, t]))
                credit_upper += float(coefficients[t]) * take
                remaining = max(0., remaining - take)
                if remaining == 0.:
                    break
        credit_upper = max(credit, float(credit_upper))
        method = "weighted_top_q_credit_lp"
    raw_bound = credit / fresh["weight_hours"]
    upper_bound = credit_upper / fresh["weight_hours"] + fresh["criticality_slack_kw"]
    return {"q": q, "R_kwh": credit_upper, "R_primal_kwh": credit,
            "credit_upper_gap_kwh": max(0., credit_upper - credit),
            "raw_bound_kw": raw_bound, "bound_kw": upper_bound,
            "q_certificate": fresh["q_certificate"],
            "count_condition_satisfied": bool(q >= fresh["q_certificate"]),
            "certifies_zero_saving": bool(upper_bound <= FEASIBILITY_TOL),
            "zero_saving_tolerance_kw": FEASIBILITY_TOL,
            "slots": fresh["slots"], "criticality_error_kw": fresh["criticality_error_kw"],
            "criticality_slack_kw": fresh["criticality_slack_kw"],
            "method": method, "elapsed_seconds": perf_counter() - start,
            "units": {"R": "kWh", "weight": "h", "bound": "kW"}}
