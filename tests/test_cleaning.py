# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Tests for src/cleaning.py. Each test builds a tiny DataFrame by hand, so the
expected answer is obvious and the tests do not need the real data file."""

import numpy as np
import pandas as pd
import pytest

from src import cleaning


def raw_rows(**overrides):
    """Return a one-row raw-format DataFrame (all text), with optional overrides."""
    row = {
        "Animal ID": "A123456", "Animal Name": "*REX", "Animal Type": "DOG",
        "Primary Color": "BRN TABBY", "Secondary Color": "", "Sex": "Male",
        "DOB": "2020-01-01", "Intake Date": "2022-01-01", "Intake Condition": "INJURED  SEVERE",
        "Intake Type": "STRAY", "Intake Subtype": "FIELD", "Reason for Intake": "",
        "Outcome Date": "2022-01-11", "Crossing": "[100 BLK MAIN ST, LONG BEACH",
        "Jurisdiction": "DISTRICT3", "Outcome Type": "ADOPTION", "Outcome Subtype": "WEB",
        "latitude": "33.80", "longitude": "-118.15", "intake_is_dead": "Alive on Intake",
        "outcome_is_dead": "FALSE", "was_outcome_alive": "1", "geopoint": "33.80, -118.15",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_rename_columns_snake_case():
    out = cleaning.rename_columns(raw_rows())
    assert "animal_id" in out.columns and "Animal ID" not in out.columns


def test_rename_columns_missing_column_raises():
    with pytest.raises(KeyError):
        cleaning.rename_columns(raw_rows().drop(columns=["Sex"]))


def test_blank_to_nan():
    out = cleaning.blank_to_nan(pd.DataFrame({"a": ["x", "", "   "]}))
    assert out["a"].isna().tolist() == [False, True, True]


def test_standardize_text_collapses_spaces_and_case():
    s = pd.Series(["INJURED  SEVERE ", np.nan])
    out = cleaning.standardize_text(s)
    assert out.iloc[0] == "injured severe" and pd.isna(out.iloc[1])


def test_standardize_color_expands_abbreviations():
    out = cleaning.standardize_color(pd.Series(["BRN TABBY", "SEAL PT"]))
    assert out.tolist() == ["brown tabby", "seal point"]


def test_standardize_categories_fixes_typos_and_districts():
    df = cleaning.rename_columns(raw_rows(**{"Intake Condition": "ILL MODERATETE"}))
    out = cleaning.standardize_categories(df)
    assert out.loc[0, "intake_condition"] == "ill moderate"
    assert out.loc[0, "jurisdiction"] == "long beach"


def test_clean_names_flags_shelter_names():
    df = pd.DataFrame({"animal_name": ["*REX", "BELLA", "*", np.nan]})
    out = cleaning.clean_names(df)
    assert out["name_given_by_shelter"].tolist() == [True, False, True, False]
    assert out["has_name"].tolist() == [True, True, False, False]
    assert out.loc[0, "animal_name"] == "Rex"


def test_clean_address_strips_leading_symbols():
    out = cleaning.clean_address(pd.Series(["[100 BLK MAIN", "`600 BLK E X"]))
    assert out.tolist() == ["100 BLK MAIN", "600 BLK E X"]


def test_fix_types_parses_and_counts_failures():
    df = cleaning.blank_to_nan(cleaning.rename_columns(raw_rows(DOB="not-a-date")))
    out, failures = cleaning.fix_types(df)
    assert pd.api.types.is_datetime64_any_dtype(out["intake_date"])
    assert failures["dob"] == 1 and pd.isna(out.loc[0, "dob"])


def test_remove_duplicates_keeps_last_and_drops_duplicate_outcome():
    df = pd.DataFrame({
        "animal_id": ["A1", "A1", "A2", "A3"],
        "intake_date": pd.to_datetime(["2022-01-01", "2022-01-01", "2022-01-01", "2022-02-01"]),
        "outcome_type": ["rescue", "adoption", "duplicate", "adoption"],
    })
    kept, removed = cleaning.remove_duplicates(df)
    assert kept["animal_id"].tolist() == ["A1", "A3"]
    assert kept.iloc[0]["outcome_type"] == "adoption"
    assert len(removed) == 2


def test_fix_invalid_values_blanks_future_dob_and_bad_coords():
    df = pd.DataFrame({
        "dob": pd.to_datetime(["2025-01-01", "2020-01-01"]),
        "intake_date": pd.to_datetime(["2022-01-01", "2022-01-01"]),
        "latitude": [45.5, 33.8], "longitude": [-73.9, -118.1],
    })
    out, counts = cleaning.fix_invalid_values(df)
    assert pd.isna(out.loc[0, "dob"]) and out.loc[1, "dob"] == pd.Timestamp("2020-01-01")
    assert np.isnan(out.loc[0, "latitude"]) and counts["coords_outside_socal_blanked"] == 1


def test_haversine_known_distance():
    # One degree of latitude is about 111.2 km.
    d = cleaning.haversine_km(np.array([0.0]), np.array([0.0]), 1.0, 0.0)
    assert d[0] == pytest.approx(111.19, abs=0.1)


def test_clean_all_builds_label_and_age():
    clean, rejected, log = cleaning.clean_all(raw_rows())
    assert clean.loc[0, "bad_outcome"] == 0
    assert clean.loc[0, "age_at_intake_years"] == pytest.approx(2.0, abs=0.01)
    assert clean.loc[0, "length_of_stay_days"] == 10
    assert log["rows_raw"] == 1 and len(rejected) == 0


def test_clean_all_euthanasia_is_bad_outcome_and_missing_is_unknown():
    df = pd.concat([raw_rows(**{"Outcome Type": "EUTHANASIA"}),
                    raw_rows(**{"Animal ID": "A999999", "Outcome Type": ""})], ignore_index=True)
    clean, _, _ = cleaning.clean_all(df)
    assert clean.loc[0, "bad_outcome"] == 1 and pd.isna(clean.loc[1, "bad_outcome"])


def test_clean_all_rejects_empty_input():
    with pytest.raises(ValueError):
        cleaning.clean_all(pd.DataFrame())


def test_load_raw_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        cleaning.load_raw(tmp_path / "nope.csv")
