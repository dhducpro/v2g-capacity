# V2G capacity under discharge withdrawals

Code for *Robust Vehicle-to-Grid Scheduling under Discharge Withdrawals: Capacity Bounds and Replenishment*.

A station commits to a charging schedule. Up to `q` vehicles may then cancel their planned discharges. The model computes the minimum station import capacity that can accommodate those cancellations. It also computes a charging-only baseline and an upper bound on the capacity savings from discharge.

## Run the six-vehicle example

From the repository root, using Python 3.12:

```sh
python -m pip install -r requirements-capacity.txt
python solve_fleet.py examples/fleet.json --q 1
python check_capacity.py
```

The example uses one synthetic fleet from the study. With one withdrawal, the charging-only baseline is **21.795159 kW**, the protected capacity is **21.191488 kW**, and the saving is **0.603671 kW**. The upper bound coincides with the saving in this example. Use any integer from 0 through 6 to explore the other budgets.

`check_capacity.py` compares the solver with explicit withdrawal enumeration and analytical two-slot solutions. It also checks interval splitting, infeasibility, and power credits on very short intervals.

## Use another fleet

Pass a JSON file with the same fields as `examples/fleet.json`. Rows in `availability` represent vehicles and columns represent time intervals.

| Field | Meaning |
| --- | --- |
| `availability` | Binary vehicle-by-interval matrix |
| `duration` | Interval lengths in hours |
| `L` | Background station load in kW |
| `Ulo`, `Uhi` | Lower and upper multipliers on net fleet power in each interval |
| `E0`, `Emin`, `Emax`, `Etarget` | Initial, minimum, maximum and required final battery energy in kWh |
| `Pc`, `Pd` | Charging and discharging limits per vehicle in kW |
| `eta_c`, `eta_d` | Charging and discharging efficiencies per vehicle |

For use from Python, `v2g_capacity.solve_capacity(model, q)` returns the solver record and the recovered schedule. `solve_reservation` computes the charging-only baseline. `reservation_cut` finds verified critical cuts, and `compute_certificate` evaluates a savings bound for a cut at a given budget. The command-line summary reports the smallest bound among the verified candidates. When no cut is available, it reports a null bound.

## Reproduce the archived study

The example and mathematical checks run from a fresh clone. The full study additionally needs the separate replication data package, including the processed charging export, selected inputs, saved schedules and result tables.

To audit those results, start with a working copy of the complete study package and overlay this repository's `replication/` directory, keeping the package's data and output files in place. From that working copy, run:

```sh
python replication/run_focused_study.py --saved-only
```

This reconstructs the inputs, audits the saved schedules and rebuilds the tables and figures. Omit `--saved-only` to rerun the focused optimization study. The study runner uses the original hash-verified solver in `replication/independent_validation/weighted_capacity.py`. The top-level module is the edited implementation for direct use.

The public repository contains the focused capacity code and one synthetic example. The full session data, manuscript, referee correspondence and supplementary economic-control package are outside this code release.
