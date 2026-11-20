# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Validation checks and the quality gate.

Each ``check_*`` function returns a boolean Series (True = row is OK) or, for
table-level checks, a single bool. ``validate`` runs them all and builds a
report; ``quality_gate`` raises ``DataQualityError`` when the data is not good
enough, so the pipeline fails loudly instead of quietly writing bad output.
"""

import pandas as pd

from src import config
from src.cleaning import ANIMAL_ID_PATTERN

REQUIRED_COLUMNS = [
    "animal_id", "animal_type", "sex", "intake_date", "intake_condition",
    "intake_type", "outcome_date", "outcome_type", "latitude", "longitude",
]
NOT_NULL_COLUMNS = ["animal_id", "animal_type", "intake_date", "intake_type", "intake_condition"]

ALLOWED_VALUES = {
    "animal_type": {"cat", "dog", "bird", "wild", "other", "rabbit", "reptile",
                    "guinea pig", "livestock", "amphibian"},
    "sex": {"male", "female", "neutered", "spayed", "unknown"},
    "intake_type": {"stray", "wildlife", "owner surrender", "welfare seized", "confiscate",
                    "return", "quarantine", "safe keep", "trap, neuter, return",
                    "euthanasia required", "foster", "adopted animal return"},
}


class DataQualityError(Exception):
    """Raised when validation fails the quality gate."""


# ------------------------------------------------------------------ table checks
def check_required_columns(df, required=REQUIRED_COLUMNS):
    """Return the list of required columns that are missing (empty list = OK)."""
    return [c for c in required if c not in df.columns]


def check_dtype_datetime(df, column):
    """True if ``column`` has a datetime dtype (i.e. dates were parsed)."""
    return pd.api.types.is_datetime64_any_dtype(df[column])


def check_dtype_numeric(df, column):
    """True if ``column`` has a numeric dtype."""
    return pd.api.types.is_numeric_dtype(df[column])


# ------------------------------------------------------------------ row checks
def check_not_null(df, column):
    """Row is OK if ``column`` is not missing."""
    return df[column].notna()


def check_unique(df, columns):
    """Row is OK if no other row has the same values in ``columns``."""
    return ~df.duplicated(subset=columns, keep=False)


def check_pattern(df, column, pattern=ANIMAL_ID_PATTERN):
    """Row is OK if the value matches the regex ``pattern`` (missing = fail)."""
    return df[column].astype("string").str.fullmatch(pattern.pattern).fillna(False).astype(bool)


def check_allowed(df, column, allowed):
    """Row is OK if the value is in ``allowed`` (missing values are allowed)."""
    return df[column].isna() | df[column].isin(allowed)


def check_range(df, column, low, high):
    """Row is OK if ``low <= value <= high`` (missing values are allowed)."""
    values = pd.to_numeric(df[column], errors="coerce")
    return values.isna() | values.between(low, high)


def check_date_order(df, earlier, later):
    """Row is OK if ``earlier <= later`` or either date is missing."""
    a, b = df[earlier], df[later]
    return a.isna() | b.isna() | (a <= b)


# ------------------------------------------------------------------ runner
def row_checks(df):
    """Return a dict ``{check_name: boolean Series}`` for every row-level rule."""
    checks = {}
    for col in NOT_NULL_COLUMNS:
        checks[f"not_null:{col}"] = check_not_null(df, col)
    checks["unique:animal_id+intake_date"] = check_unique(df, ["animal_id", "intake_date"])
    checks["pattern:animal_id"] = check_pattern(df, "animal_id")
    for col, allowed in ALLOWED_VALUES.items():
        checks[f"allowed:{col}"] = check_allowed(df, col, allowed)
    checks["range:latitude"] = check_range(df, "latitude", *config.LAT_RANGE)
    checks["range:longitude"] = check_range(df, "longitude", *config.LON_RANGE)
    checks["order:dob<=intake_date"] = check_date_order(df, "dob", "intake_date")
    checks["order:intake_date<=outcome_date"] = check_date_order(df, "intake_date", "outcome_date")
    if "age_at_intake_years" in df.columns:
        checks["range:age_at_intake_years"] = check_range(df, "age_at_intake_years", 0, config.MAX_AGE_YEARS)
    return checks


def validate(df):
    """Run all checks.

    Returns a dict with:
    * ``checks``  - DataFrame: one row per check with failed-row count
    * ``row_pass_rate`` - share of rows that pass *every* row-level check
    * ``critical_failures`` - list of messages for checks that must never fail
    * ``row_ok`` - boolean Series marking rows that passed everything
    """
    missing = check_required_columns(df)
    if missing:
        # Without these columns the other checks cannot even run.
        raise DataQualityError(f"Missing required columns: {missing}")

    critical = []
    if not check_dtype_datetime(df, "intake_date"):
        critical.append("intake_date is not a datetime column")
    if not check_dtype_numeric(df, "latitude"):
        critical.append("latitude is not numeric")

    results = row_checks(df)
    rows = []
    for name, ok in results.items():
        rows.append({"check": name, "rows_failed": int((~ok).sum()),
                     "pct_failed": round(100 * (~ok).mean(), 3), "passed": bool(ok.all())})
    table = pd.DataFrame(rows)
    row_ok = pd.concat(results, axis=1).all(axis=1) if results else pd.Series(True, index=df.index)

    for name in ("not_null:animal_id", "not_null:intake_date"):
        if not results[name].all():
            critical.append(f"{name} failed on {int((~results[name]).sum())} rows")

    return {
        "checks": table,
        "row_pass_rate": float(row_ok.mean()) if len(df) else 0.0,
        "checks_passed": int(table["passed"].sum()),
        "checks_total": len(table),
        "critical_failures": critical,
        "row_ok": row_ok,
    }


def quality_gate(report, threshold=config.VALIDATION_PASS_THRESHOLD):
    """Stop the pipeline if the data is not trustworthy.

    Fails when any critical check failed or the row pass rate is below
    ``threshold``. The error message lists the worst checks so the user knows
    what to fix.
    """
    rate = report["row_pass_rate"]
    if report["critical_failures"] or rate < threshold:
        worst = (report["checks"].sort_values("rows_failed", ascending=False)
                 .head(5)[["check", "rows_failed"]].to_dict("records"))
        raise DataQualityError(
            f"QUALITY GATE FAILED: row pass rate {rate:.2%} (threshold {threshold:.0%}); "
            f"critical: {report['critical_failures'] or 'none'}; worst checks: {worst}"
        )
    return True
