# V2G Capacity

Station capacity under EV discharge withdrawals, using [Caltech ACN-Data](https://ev.caltech.edu/dataset) and synthetic fleets.

Download the [study version](https://github.com/dhducpro/v2g-capacity/archive/refs/tags/v1.0.zip), open [experiments.ipynb](experiments.ipynb) and select **Run All**. The [capacity data](study-data.zip) are included. Results go to `runs/`.

The paper supplies the economic, calibration and runtime records separately as `reviewer-control-records.zip`.

For the six-vehicle example:

```sh
python -m pip install -r requirements-capacity.txt
python solve_fleet.py examples/fleet.json --q 1
```
