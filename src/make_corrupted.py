# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Create a deliberately corrupted COPY of the raw data to prove the quality gate works.

The real raw file is only read, never written. The copy goes to data/corrupted/.
    python -m src.make_corrupted
    python -m src.pipeline_a --input data/corrupted/shelter_corrupted.csv \
        --metrics-out outputs/reports/gate_demo/metrics_corrupted.json      # -> exits with code 1
"""

import numpy as np

from src import cleaning, config

OUT_FILE = config.ROOT / "data" / "corrupted" / "shelter_corrupted.csv"


def corrupt(raw, fraction=0.10, seed=config.RANDOM_STATE):
    """Return a corrupted copy: in ``fraction`` of rows inject one of three faults.

    * an unknown animal type (``DRAGON``)
    * an outcome date years *before* the intake date
    * an animal ID in the wrong format
    """
    rng = np.random.default_rng(seed)
    bad = raw.copy()
    idx = rng.choice(len(bad), size=int(fraction * len(bad)), replace=False)
    parts = np.array_split(idx, 3)
    bad.loc[parts[0], "Animal Type"] = "DRAGON"
    bad.loc[parts[1], "Outcome Date"] = "2010-01-01"
    bad.loc[parts[2], "Animal ID"] = "BAD-ID"
    return bad


def main():
    """Write the corrupted copy and print where it went."""
    raw = cleaning.load_raw()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    corrupt(raw).to_csv(OUT_FILE, index=False)
    print(f"Corrupted copy written to {OUT_FILE}")


if __name__ == "__main__":
    main()
