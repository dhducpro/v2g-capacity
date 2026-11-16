"""Independent post-run implementation audit on a fixed input-only sample.

This audit is NOT an extra statistical validation cohort. For each N=6,12,24,
it selects the lexicographically first observed case and the lexicographically
first synthetic constant-profile case. It compares q=0,1,N capacities with the
historical explicit-withdrawal/prefix-energy LP, then compares q=1 cut credit
with a separate LP that enumerates all single-vehicle withdrawal rows.

The selected frozen case files and both frozen solver modules are read-only.
Every discrepancy/error is retained in the JSON result. No selection or input
adjustment is made based on results.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cplex_lp import linprog
from scipy.sparse import coo_matrix

from weighted_capacity import compute_certificate, reservation_cut, solve_capacity

HERE = Path(__file__).resolve().parent
REPLICATION = HERE.parent
OLD_SOURCE = REPLICATION / "sensitivity_checks/time_guard_sensitivity.py"
NEW_SOURCE = HERE / "weighted_capacity.py"
TOLERANCE_KW = 1e-6


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(path):
    model = json.loads(path.read_text())
    for key in ("availability", "duration", "L", "E0", "Emin", "Emax", "Etarget",
                "Pc", "Pd", "eta_c", "eta_d", "Ulo", "Uhi", "boundaries_seconds"):
        if key in model:
            model[key] = np.asarray(model[key], dtype=float)
    return model


def choose_inputs():
    selected = []
    for folder in ("observed_cases", "synthetic_cases"):
        candidates = sorted((HERE / "inputs" / folder).glob("*.json"))
        for n in (6, 12, 24):
            matching = []
            for path in candidates:
                case = json.loads(path.read_text())
                if case["N"] == n and case["metadata"]["profile"] == "constant":
                    matching.append(path)
            if not matching:
                raise ValueError(f"No constant-profile case for {folder}, N={n}")
            selected.append(matching[0])
    return selected


def explicit_credit(model, slots):
    """Max weighted surviving credit using explicit q=1 cancellation rows.

    Variables are nonnegative grid-side discharge powers z_it and surviving
    power v_t. For each canceled vehicle j, v_t <= sum_{i != j} z_it.
    All donor-energy quantities are independently reconstructed from inputs.
    No top-q epigraph, row builder, or stored donor budget is reused.
    """
    a = model["availability"]
    n, _ = a.shape
    slots = np.asarray(slots, dtype=int)
    outside = np.ones(a.shape[1], dtype=bool)
    outside[slots] = False
    h = (model["Etarget"] - model["E0"]) / model["eta_c"]
    outside_charge = np.sum(a[:, outside] * model["Pc"][:, None]
                            * model["duration"][None, outside], axis=1)
    donor_energy = np.minimum(model["eta_d"] * (model["Emax"] - model["Etarget"]),
        model["eta_c"] * model["eta_d"] * np.maximum(0., outside_charge - h))
    donor_energy = np.maximum(donor_energy, 0.)
    dt = model["duration"][slots]
    pd = a[:, slots] * model["Pd"][:, None]
    nt, nz = len(slots), n * len(slots)
    nv = nz + nt
    ri, cj, vv, rhs = [], [], [], []
    for i in range(n):
        row = len(rhs)
        ri.extend([row] * nt)
        cj.extend(range(i * nt, (i + 1) * nt))
        vv.extend(dt.tolist()); rhs.append(float(donor_energy[i]))
    for withdrawn in range(n):
        survivors = [i for i in range(n) if i != withdrawn]
        for t in range(nt):
            row = len(rhs)
            ri.extend([row] * (n + 0))
            cj.extend([nz + t] + [i * nt + t for i in survivors])
            vv.extend([1.] + [-1.] * len(survivors)); rhs.append(0.)
    matrix = coo_matrix((vv, (ri, cj)), shape=(len(rhs), nv)).tocsr()
    objective = np.r_[np.zeros(nz), -dt]
    bounds = [(0., float(v)) for v in pd.ravel()] + [(0., None)] * nt
    start = perf_counter()
    result = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=bounds)
    elapsed = perf_counter() - start
    if not result.success:
        return {"status":"solver_error", "message":str(result.message),
                "R_kwh":None, "elapsed_seconds":elapsed}
    z = result.x[:nz].reshape(n, nt)
    survivor = np.minimum.reduce([z.sum(axis=0) - z[i] for i in range(n)])
    direct = float(np.dot(dt, survivor))
    primal = -float(result.fun)
    energy_error = max(0., float(np.max(z @ dt - donor_energy)))
    power_error = max(0., float(np.max(-z)), float(np.max(z - pd)),
                      float(np.max(result.x[nz:] - survivor)))
    return {"status":"optimal", "R_kwh":primal, "direct_R_kwh":direct,
            "direct_objective_error_kwh":abs(primal-direct),
            "energy_residual_kwh":energy_error, "power_residual_kw":power_error,
            "elapsed_seconds":elapsed, "withdrawal_rows":n * nt,
            "critical_weight_hours":float(np.sum(dt/model["Uhi"][slots]))}


def main():
    spec = importlib.util.spec_from_file_location("historical_enumerated_capacity", OLD_SOURCE)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    selected = choose_inputs()
    frozen = json.loads((HERE / "validation_protocol.json").read_text())
    sources = {"independent_validation/weighted_capacity.py":digest(NEW_SOURCE),
               "sensitivity_checks/time_guard_sensitivity.py":digest(OLD_SOURCE)}
    for name, value in sources.items():
        if frozen["source_sha256"][name] != value:
            raise AssertionError(f"Frozen source hash mismatch: {name}")
    selected_records = [{"path":p.relative_to(HERE).as_posix(),"sha256":digest(p)} for p in selected]
    for record in selected_records:
        if frozen["input_sha256"][record["path"]] != record["sha256"]:
            raise AssertionError(f"Frozen input hash mismatch: {record['path']}")
    plan = {"study":"Post-run independent implementation audit, not additional statistical validation",
            "selection_rule":"Lexicographically first observed constant-profile case and first synthetic constant-profile case for each N=6,12,24; no outcome-based choice",
            "q_values":"0,1,N for each selected case",
            "credit_check":"q=1 with explicit singleton cancellation rows on the selected input-only critical cut",
            "frozen_before_audit_solves_utc":datetime.now(timezone.utc).isoformat(),
            "script_sha256":digest(Path(__file__)), "sources_sha256":sources,
            "selected_inputs":selected_records,"capacity_tolerance_kw":TOLERANCE_KW}
    plan_path = HERE / "generalized_implementation_audit_protocol.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if existing["script_sha256"] != plan["script_sha256"]:
            revision = json.loads((HERE / "audit_source_revision.json").read_text())
            if (existing["script_sha256"], plan["script_sha256"]) != (revision["archived_sha256"], revision["portable_sha256"]):
                raise AssertionError("Unrecognized audit source revision")
        for key in ("sources_sha256", "selected_inputs"):
            if existing[key] != plan[key]:
                raise AssertionError(f"Audit protocol changed: {key}; preserve it and record an amendment")
        plan = existing
    else:
        plan_path.write_text(json.dumps(plan,indent=2)+"\n")
    comparisons, credits, errors = [], [], []
    for path in selected:
        model = load_case(path)
        if not np.all(model["Ulo"] == 1.) or not np.all(model["Uhi"] == 1.):
            raise AssertionError("Historical comparison requires explicit constant unit coefficients")
        for q in (0, 1, model["N"]):
            row = {"case":path.stem,"N":model["N"],"T":model["T"],"q":q}
            try:
                modern, _ = solve_capacity(model,q)
                historical, _ = old.solve(model,q,guard=True,endpoints=(1.,))
                row.update(weighted=modern,explicit_prefix=historical,
                    status_agreement=modern["status"] == historical["status"])
                if modern["status"] == historical["status"] == "optimal":
                    row["capacity_error_kw"] = abs(modern["M_kw"]-historical["M_kw"])
                    row["pass"] = row["capacity_error_kw"] <= TOLERANCE_KW
                else:
                    row["capacity_error_kw"] = None
                    row["pass"] = row["status_agreement"]
            except Exception as error:
                row.update(pass_check=False,error=repr(error))
                row["pass"] = False
                errors.append({"case":path.stem,"q":q,"error":repr(error)})
            comparisons.append(row)
        credit_row = {"case":path.stem,"N":model["N"],"q":1}
        try:
            extraction = reservation_cut(model)
            credit_row["cut_status"] = extraction["status"]
            cut = extraction["selected_cut"]
            if cut is None:
                credit_row.update(status="unavailable_cut_retained",pass_check=True)
            else:
                modern_credit = compute_certificate(model,cut,1)
                independent = explicit_credit(model,cut["slots"])
                weight = independent["critical_weight_hours"]
                if independent["status"] != "optimal":
                    raise RuntimeError(independent["message"])
                primal_error = abs(modern_credit["R_primal_kwh"]-independent["R_kwh"])/weight
                upper_violation = max(0.,independent["R_kwh"]-modern_credit["R_kwh"])/weight
                credit_row.update(status="compared",slots=cut["slots"],weighted=modern_credit,
                    explicit_withdrawal=independent,primal_credit_error_kw=primal_error,
                    conservative_upper_violation_kw=upper_violation,
                    pass_check=primal_error<=TOLERANCE_KW and upper_violation<=TOLERANCE_KW)
        except Exception as error:
            credit_row.update(status="error",pass_check=False,error=repr(error))
            errors.append({"case":path.stem,"credit_q":1,"error":repr(error)})
        credits.append(credit_row)
        print(json.dumps({"case":path.stem,"capacity_checks":3,"credit_status":credit_row["status"]}),flush=True)
    unchanged = (all(digest(REPLICATION/name)==value for name,value in sources.items())
                 and all(digest(HERE/r["path"])==r["sha256"] for r in selected_records))
    report = {"status":"PASS" if unchanged and not errors and all(r["pass"] for r in comparisons)
              and all(r["pass_check"] for r in credits) else "FAIL",
              "scope":plan["study"],"completed_utc":datetime.now(timezone.utc).isoformat(),
              "protocol":plan,"executed_audit_sha256":digest(Path(__file__)),
              "frozen_sources_and_inputs_unchanged":unchanged,
              "capacity_comparison_pairs":len(comparisons),
              "credit_comparisons":sum(r["status"]=="compared" for r in credits),
              "unavailable_cuts_retained":sum(r["status"]=="unavailable_cut_retained" for r in credits),
              "max_capacity_error_kw":max((r.get("capacity_error_kw") or 0.) for r in comparisons),
              "max_credit_primal_error_kw":max((r.get("primal_credit_error_kw") or 0.) for r in credits),
              "max_credit_upper_violation_kw":max((r.get("conservative_upper_violation_kw") or 0.) for r in credits),
              "capacity_comparisons":comparisons,"credit_checks":credits,"errors":errors}
    (HERE/"generalized_implementation_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(dict,list))},indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
