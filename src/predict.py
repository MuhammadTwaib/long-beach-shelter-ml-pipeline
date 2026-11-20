# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Score new intake records with the saved pipeline.

    python -m src.predict data/new/unseen_intakes.csv
    python -m src.predict my_rows.csv --output my_predictions.csv

The input CSV needs the columns in ``features.INPUT_COLUMNS`` (extra columns
are kept in the output but not used). Output = input + ``risk_probability`` +
``high_risk_flag`` (1 = likely not to leave alive -> prioritise vet / rescue).
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src import config


class InputError(ValueError):
    """Raised when the CSV to score is missing columns or has unusable values."""


def load_bundle(path=config.MODEL_FILE):
    """Load the saved pipeline + threshold; clear message if training was not run."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No model at {path}. Train it first: python -m src.train")
    return joblib.load(path)


def validate_input(df, required):
    """Check the new rows before scoring. Raises ``InputError`` with every problem found."""
    problems = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if df.empty:
        problems.append("the file has no rows")
    if "intake_date" in df.columns:
        bad_dates = pd.to_datetime(df["intake_date"], format="%Y-%m-%d", errors="coerce").isna()
        if bad_dates.any():
            problems.append(f"{int(bad_dates.sum())} rows have a missing or invalid intake_date "
                            f"(row numbers: {list(df.index[bad_dates][:10])})")
    if problems:
        raise InputError("Cannot score file: " + "; ".join(problems))


def predict_frame(df, bundle):
    """Return a copy of ``df`` with risk probability and the yes/no flag added."""
    validate_input(df, bundle["input_columns"])
    proba = bundle["pipeline"].predict_proba(df)[:, 1]
    out = df.copy()
    out["risk_probability"] = proba.round(4)
    out["high_risk_flag"] = (proba >= bundle["threshold"]).astype(int)
    return out


def main(argv=None):
    """CLI entry point. Exit code 0 = success, 2 = bad input."""
    parser = argparse.ArgumentParser(description="Score new shelter intakes")
    parser.add_argument("input_csv", type=Path, help="CSV of new intake rows")
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write predictions (default: <input>_predictions.csv)")
    parser.add_argument("--model", type=Path, default=config.MODEL_FILE)
    args = parser.parse_args(argv)

    if not args.input_csv.exists():
        print(f"Input file not found: {args.input_csv}", file=sys.stderr)
        return 2
    output = args.output or args.input_csv.with_name(args.input_csv.stem + "_predictions.csv")
    bundle = load_bundle(args.model)
    rows = pd.read_csv(args.input_csv)
    try:
        scored = predict_frame(rows, bundle)
    except InputError as err:
        print(err, file=sys.stderr)
        return 2
    scored.to_csv(output, index=False)
    print(f"Scored {len(scored)} rows with '{bundle['model_name']}' (threshold {bundle['threshold']:.3f}); "
          f"{int(scored['high_risk_flag'].sum())} flagged high-risk -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
