# V2G Capacity

Install IBM ILOG CPLEX Studio and set `CPLEX_BIN` to its executable (or `CPLEX_LIBRARY` to its shared library). Open [experiments.ipynb](experiments.ipynb) and select **Run All**. Inputs are included in [study-data.zip](study-data.zip). Each run writes its own schedules, tables and figures under `runs/`.

[Protected capacities](results/cplex_frontiers.csv) for the Caltech and synthetic fleets were solved with CPLEX 22.1.2. The paper provides its economic and calibration experiments in a separate supplementary archive.
