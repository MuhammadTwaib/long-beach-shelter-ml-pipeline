# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Tests for src/validation.py - every check is shown failing on bad data."""

import pandas as pd
import pytest

from src import validation


def good_frame(n=3):
    """A small, fully valid cleaned-format frame."""
    return pd.DataFrame({
        "animal_id": [f"A00000{i}" for i in range(n)],
        "animal_type": ["cat"] * n,
        "sex": ["male"] * n,
        "dob": pd.to_datetime(["2020-01-01"] * n),
        "intake_date": pd.to_datetime(["2022-01-01"] * n),
        "intake_condition": ["normal"] * n,
        "intake_type": ["stray"] * n,
        "outcome_date": pd.to_datetime(["2022-01-05"] * n),
        "outcome_type": ["adoption"] * n,
        "latitude": [33.8] * n,
        "longitude": [-118.1] * n,
        "age_at_intake_years": [2.0] * n,
    })


def test_required_columns_detects_missing():
    assert validation.check_required_columns(good_frame().drop(columns=["sex"])) == ["sex"]


def test_not_null():
    df = good_frame()
    df.loc[1, "animal_type"] = None
    assert validation.check_not_null(df, "animal_type").tolist() == [True, False, True]


def test_unique_marks_both_copies():
    df = good_frame()
    df.loc[2, "animal_id"] = df.loc[1, "animal_id"]
    assert validation.check_unique(df, ["animal_id", "intake_date"]).tolist() == [True, False, False]


def test_pattern():
    df = good_frame()
    df.loc[0, "animal_id"] = "X1"
    assert not validation.check_pattern(df, "animal_id").iloc[0]


def test_allowed_values():
    df = good_frame()
    df.loc[0, "animal_type"] = "dragon"
    assert validation.check_allowed(df, "animal_type", {"cat"}).tolist() == [False, True, True]


def test_range():
    df = good_frame()
    df.loc[0, "latitude"] = 99.0
    assert not validation.check_range(df, "latitude", 30, 40).iloc[0]


def test_date_order():
    df = good_frame()
    df.loc[0, "outcome_date"] = pd.Timestamp("2021-01-01")
    assert not validation.check_date_order(df, "intake_date", "outcome_date").iloc[0]


def test_dtype_checks():
    df = good_frame()
    assert validation.check_dtype_datetime(df, "intake_date")
    assert not validation.check_dtype_numeric(df.astype({"latitude": str}), "latitude")


def test_validate_good_data_passes_gate():
    report = validation.validate(good_frame())
    assert report["row_pass_rate"] == 1.0
    assert validation.quality_gate(report, threshold=0.95)


def test_validate_missing_required_column_raises():
    with pytest.raises(validation.DataQualityError):
        validation.validate(good_frame().drop(columns=["intake_date"]))


def test_gate_fails_loudly_on_bad_rows():
    df = good_frame(10)
    df.loc[:2, "animal_type"] = "dragon"          # 30% bad rows
    report = validation.validate(df)
    assert report["row_pass_rate"] == pytest.approx(0.7)
    with pytest.raises(validation.DataQualityError, match="QUALITY GATE FAILED"):
        validation.quality_gate(report, threshold=0.95)


def test_gate_fails_on_critical_null_id_even_if_rate_ok():
    df = good_frame(100)
    df.loc[0, "animal_id"] = None                 # only 1% bad, but critical
    report = validation.validate(df)
    with pytest.raises(validation.DataQualityError):
        validation.quality_gate(report, threshold=0.5)
