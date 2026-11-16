# V2G capacity under discharge withdrawals

This code computes the minimum station import capacity when up to `q` vehicles cancel their planned discharges and charging stays fixed.

The study uses charging sessions from [Caltech ACN-Data](https://ev.caltech.edu/dataset) and synthetic fleets. The example uses six simulated vehicles. Full replication requires the separate study data package.

## Run the example

With Python 3.12, run:

```sh
python -m pip install -r requirements-capacity.txt
python solve_fleet.py examples/fleet.json --q 1
python check_capacity.py
```
