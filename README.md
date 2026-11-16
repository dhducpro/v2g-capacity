# V2G Capacity

Station capacity under EV discharge withdrawals, using [Caltech ACN-Data](https://ev.caltech.edu/dataset) and synthetic fleets.

Open [experiments.ipynb](experiments.ipynb), place the separate `study-data.zip` beside it, and select **Run All**. Results, figures and logs are saved under `runs/`.

For the six-vehicle example:

```sh
python -m pip install -r requirements-capacity.txt
python solve_fleet.py examples/fleet.json --q 1
```
