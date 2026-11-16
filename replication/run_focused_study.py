"""Verify the archived study inputs and rebuild outputs from the replication package."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLICATION = ROOT / "replication"

INPUT_CHECKS = [
    ("cohort_reconstruction/reconstruct_cohort.py",),
    ("independent_validation/build_independent_inputs.py", "--check"),
]

OPTIMIZATION_STEPS = [
    ("capacity_checks/build_hourly_contexts.py",),
    ("capacity_checks/critical_cut_audit.py",),
    ("capacity_checks/critical_cut_edge_checks.py",),
    ("capacity_checks/full_lp_checks.py",),
    ("capacity_checks/sharp_class_independent_check.py",),
    ("sensitivity_checks/time_guard_sensitivity.py", "--run"),
    ("independent_validation/run_validation.py", "--run"),
    ("independent_validation/contract_controls.py",),
    ("independent_validation/check_chronological_counterexample.py",),
    ("independent_validation/check_weighted_capacity.py",),
    ("independent_validation/audit_generalized_implementation.py",),
]

OUTPUT_CHECKS = [
    ("sensitivity_checks/audit_saved_schedules.py",),
    ("independent_validation/run_validation.py", "--audit-saved"),
    ("certificate_comparison.py",),
    ("build_capacity_outputs.py",),
    ("build_capacity_mechanism.py",),
    ("build_revision_sensitivity_outputs.py",),
    ("build_publication_tables.py",),
    ("build_presentation_figures.py",),
    ("build_validation_outputs.py",),
    ("independent_validation/summarize_savings.py",),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saved-only",
        action="store_true",
        help="Audit saved schedules and rebuild outputs without rerunning the optimization.",
    )
    args = parser.parse_args()
    if not (REPLICATION / "cohort_reconstruction/inputs/caltech.csv").is_file():
        parser.exit(1, "This workflow needs the study data package. See the README.\n")

    steps = INPUT_CHECKS + ([] if args.saved_only else OPTIMIZATION_STEPS) + OUTPUT_CHECKS
    for script, *options in steps:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, REPLICATION / script, *options], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
