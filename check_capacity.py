"""Check the release solver against independent enumeration and analytical cases."""

import v2g_capacity
from replication.independent_validation.check_weighted_capacity import run


if __name__ == "__main__":
    run(v2g_capacity)
