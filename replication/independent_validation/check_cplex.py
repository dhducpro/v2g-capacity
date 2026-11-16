"""Recompute the protected capacity frontiers with the CPLEX Interactive Optimizer."""
import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from weighted_capacity import _model, audit_schedule


def expression(terms):
    return " ".join(f"{v:+.17g} {name}" for v, name in terms if v)


def write_lp(path, m, q):
    rows = []
    bounds = ["0 <= peak"]
    active = np.argwhere(m["availability"]).tolist()
    for i in range(m["N"]):
        times = np.flatnonzero(m["availability"][i])
        if not len(times):
            if m["E0"][i] < m["Etarget"][i]:
                raise ValueError("Disconnected vehicle cannot meet its target")
            continue
        guard = []
        for j, t in enumerate(times):
            key = f"{i}_{t}"
            c, d, e = "ch_"+key, "dis_"+key, "soc_"+key
            dt = m["duration"][t]
            charge = dt * m["eta_c"][i]
            terms = [(1,e), (-charge,c), (dt/m["eta_d"][i],d)]
            rhs = m["E0"][i]
            if j:
                terms.append((-1,f"soc_{i}_{times[j-1]}"))
                rhs = 0
            rows.append(f"{expression(terms)} = {rhs:.17g}")
            lower = m["Emin"][i] if j+1 < len(times) else max(m["Emin"][i],m["Etarget"][i])
            bounds.extend([f"0 <= {c} <= {m['Pc'][i]:.17g}",
                           f"0 <= {d} <= {(0 if q == m['N'] else m['Pd'][i]):.17g}",
                           f"{lower:.17g} <= {e} <= {m['Emax'][i]:.17g}"])
            guard.append((charge,c))
        rows.append(f"{expression(guard)} <= {m['Emax'][i]-m['E0'][i]:.17g}")
    for t in range(m["T"]):
        vehicles = np.flatnonzero(m["availability"][:,t])
        demand = [(1,f"ch_{i}_{t}") for i in vehicles]
        if q < len(vehicles):
            demand.extend((-1,f"dis_{i}_{t}") for i in vehicles)
            if q:
                demand.append((q,f"theta_{t}"))
                for i in vehicles:
                    excess = f"excess_{i}_{t}"
                    rows.append(f"dis_{i}_{t} - theta_{t} - {excess} <= 0")
                    demand.append((1,excess))
        for u in sorted({m["Ulo"][t],m["Uhi"][t]}):
            terms = [(-1,"peak")] + [(u*v,name) for v,name in demand]
            rows.append(f"{expression(terms)} <= {-m['L'][t]:.17g}")
    path.write_text("Minimize\n peak\nSubject To\n"+"\n".join(rows)+"\nBounds\n"+
                    "\n".join(bounds)+"\nEnd\n", encoding="ascii")
    return active


def write_credit_lp(path, m, cut, q):
    budgets = cut["donor_budget_kwh"]
    objective, rows, bounds = [], [], []
    energy = [[] for _ in budgets]
    for t in cut["slots"]:
        donors = [i for i, budget in enumerate(budgets)
                  if budget > 0 and m["availability"][i,t] and m["Pd"][i] > 0]
        if len(donors) <= q:
            continue
        dt = m["duration"][t]
        terms = [(1,f"net_{t}")]
        objective.append((dt,f"net_{t}"))
        if q:
            terms.append((q,f"theta_{t}"))
        for i in donors:
            z = f"donor_{i}_{t}"
            energy[i].append((dt,z))
            terms.append((-1,z))
            bounds.append(f"0 <= {z} <= {m['Pd'][i]:.17g}")
            if q:
                excess = f"excess_{i}_{t}"
                rows.append(f"{z} - theta_{t} - {excess} <= 0")
                terms.append((1,excess))
        rows.append(expression(terms)+" <= 0")
    for terms,budget in zip(energy,budgets):
        if terms:
            rows.append(f"{expression(terms)} <= {budget:.17g}")
    if not objective:
        objective = [(1,"credit")]
        rows = ["credit = 0"]
    path.write_text("Maximize\n"+expression(objective)+"\nSubject To\n"+"\n".join(rows)+
                    "\nBounds\n"+"\n".join(bounds)+"\nEnd\n",encoding="ascii")


def solve_fleet(task):
    binary, fleet, reference, evidence = task
    m = _model(fleet)
    records = []
    with tempfile.TemporaryDirectory(prefix="v2g-cplex-") as folder:
        work = Path(folder)
        commands = ["set threads 1", "set simplex tolerances feasibility 1e-9",
                    "set simplex tolerances optimality 1e-9"]
        for q in range(m["N"] + 1):
            active = write_lp(work / f"{q}.lp", m, q)
            commands.extend([f"read {q}.lp", "optimize", f"write {q}.sol"])
        cuts = evidence["cut_extraction"]["cuts"]
        for cert in evidence["certificates"]:
            for j,cut in enumerate(cuts):
                stem = f"credit_{cert['q']}_{j}"
                write_credit_lp(work / f"{stem}.lp", m, cut, cert["q"])
                commands.extend([f"read {stem}.lp", "optimize", f"write {stem}.sol"])
        run = subprocess.run([binary,"-c",*commands],cwd=work,capture_output=True,text=True,
                             errors="replace",check=True)
        version = re.search(r"Version identifier:\s*([\d.]+)",run.stdout)
        for q in range(m["N"] + 1):
            path = work / f"{q}.sol"
            if not path.exists():
                raise RuntimeError(f"{m['name']} q={q}: {run.stdout[-1800:]} {run.stderr[-800:]}")
            sol = ET.parse(path)
            header = sol.find("header").attrib
            if header["solutionStatusValue"] != "1":
                raise RuntimeError(f"{m['name']} q={q}: {header['solutionStatusString']}")
            values = {x.get("name"):float(x.get("value")) for x in sol.findall("variables/variable")}
            peak = float(header["objectiveValue"])
            c = np.zeros((m["N"],m["T"]))
            d = np.zeros_like(c)
            for i,t in active:
                c[i,t] = values.get(f"ch_{i}_{t}",0.)
                d[i,t] = values.get(f"dis_{i}_{t}",0.)
            k = (m["eta_c"]*m["eta_d"])[:,None]
            cycle = np.minimum(c,d/k)
            audit = audit_schedule(m,q,c-cycle,d-k*cycle,peak)
            diff = abs(peak-reference[q])
            if max(diff,audit["energy_residual_kwh"],audit["power_residual_kw"],audit["overlap_kw"]) > 1e-6:
                raise ValueError((m["name"],q,diff,audit))
            cert = evidence["certificates"][q]
            expected = {tuple(v["slots"]): v["R_primal_kwh"] for v in cert["candidates"]}
            credit_errors = []
            for j,cut in enumerate(cuts):
                result = ET.parse(work / f"credit_{q}_{j}.sol").find("header").attrib
                if result["solutionStatusValue"] != "1":
                    raise ValueError((m["name"], q, result))
                error = abs(float(result["objectiveValue"])-expected[tuple(cut["slots"])])
                if error > 1e-6:
                    raise ValueError((m["name"], q, "cut credit", error))
                credit_errors.append(error)
            records.append(dict(credit_lps=len(cuts),credit_error_kwh=max(credit_errors),case=m["name"],q=q,status=header["solutionStatusString"],
                                capacity_kw=peak,reference_kw=reference[q],difference_kw=diff,
                                energy_residual_kwh=audit["energy_residual_kwh"],
                                power_residual_kw=audit["power_residual_kw"],overlap_kw=audit["overlap_kw"],
                                cplex_version=version.group(1) if version else "unknown"))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cplex", default=os.getenv("CPLEX_BIN","cplex"))
    parser.add_argument("--data",type=Path,default=Path("study-data.zip"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=1)
    args = parser.parse_args()
    binary = shutil.which(args.cplex)
    if not binary:
        parser.error("Set CPLEX_BIN or pass --cplex with the path to the optimizer.")
    prefix = "replication/independent_validation/"
    with ZipFile(args.data) as archive:
        references = {}
        for row in csv.DictReader(io.StringIO(archive.read(prefix+"results/frontiers.csv").decode())):
            if row["status"] != "optimal":
                raise ValueError(row)
            references.setdefault(row["case"],{})[int(row["q"])] = float(row["M_kw"])
        inputs = [json.loads(archive.read(n)) for n in sorted(archive.namelist())
                  if n.startswith(prefix+"inputs/") and "_cases/" in n and n.endswith(".json")]
        evidence = {m["name"]:json.loads(archive.read(prefix+"results/cases/"+m["name"]+".json")) for m in inputs}
    if {m["name"] for m in inputs} != references.keys() or len(inputs)!=len(references):
        raise ValueError("Fleet list differs from the archived results")
    tasks = [(binary,m,references[m["name"]],evidence[m["name"]]) for m in inputs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for group in pool.map(solve_fleet,tasks) for row in group]
    if len(rows) != sum(map(len,references.values())):
        raise ValueError("Incomplete frontier")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CPLEX {rows[0]['cplex_version']}: {len(inputs)} fleets, {len(rows)} optimal capacities")
    print(f"Maximum difference: {max(r['difference_kw'] for r in rows):.9g} kW")


if __name__ == "__main__":
    main()
