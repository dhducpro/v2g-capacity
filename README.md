# V2G Capacity

Install IBM ILOG CPLEX Studio and set `CPLEX_BIN` to its executable (or `CPLEX_LIBRARY` to its shared library). Open [experiments.ipynb](experiments.ipynb) and select **Run All**. Inputs are included in [study-data.zip](study-data.zip). Each run writes its own schedules, tables and figures under `runs/`.

[Protected capacities](results/cplex_frontiers.csv) for the Caltech and synthetic fleets were solved with CPLEX 22.1.2. The [replication archive](reviewer-control-records.zip) contains the code, inputs and results for the economic boundary tests, calibration diagnostics and runtime comparisons in Appendix B.1–B.3 of the manuscript. After extraction, follow `replication/reviewer_control_records/README.txt` to reproduce the experiments.
