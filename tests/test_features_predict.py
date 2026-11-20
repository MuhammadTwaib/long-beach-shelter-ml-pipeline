# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Tests for feature building, leakage rules and predict.py input validation.

The model-dependent tests are skipped if the model has not been trained yet."""

import numpy as np
import pandas as pd
import pytest

from src import config, features, predict


def new_rows():
    return pd.DataFrame({
        "animal_type": ["DOG", "cat"], "primary_color": ["black", "never seen colour"],
        "secondary_color": [np.nan, "white"], "sex": ["Neutered", "female"],
        "dob": ["2020-01-01", "2030-01-01"], "intake_date": ["2024-12-31", "2025-01-01"],
        "intake_condition": ["normal", "ill severe"], "intake_type": ["stray", "stray"],
        "intake_subtype": ["field", np.nan], "reason_for_intake": [np.nan, "move"],
        "jurisdiction": ["long beach", "cerritos"], "latitude": [33.81, np.nan],
        "longitude": [-118.12, np.nan],
    })


def test_no_leaky_column_is_a_feature():
    assert not set(features.FEATURES) & set(features.LEAKY_COLUMNS)


def test_build_features_shapes_and_values():
    out = features.build_features(new_rows())
    assert list(out.columns) == features.FEATURES
    assert out.loc[0, "age_at_intake_years"] == pytest.approx(5.0, abs=0.01)
    assert np.isnan(out.loc[1, "age_at_intake_years"])        # DOB in the future -> unknown
    assert out.loc[0, "sex_simple"] == "male"                   # neutered -> male (no altered status)
    assert out.loc[0, "animal_type"] == "dog"                   # lower-cased
    assert out.loc[0, "distance_km"] < 1                        # next to the shelter
    assert out.loc[1, "reason_given"] == 1


def test_month_encoding_is_circular():
    df = pd.concat([new_rows().iloc[[0]]] * 2, ignore_index=True)
    df["intake_date"] = ["2024-12-15", "2024-01-15"]
    out = features.build_features(df)
    dec, jan = out.loc[0, ["intake_month_sin", "intake_month_cos"]], out.loc[1, ["intake_month_sin", "intake_month_cos"]]
    assert np.linalg.norm(dec.to_numpy(float) - jan.to_numpy(float)) < 0.6


def test_build_features_missing_column_raises():
    with pytest.raises(KeyError):
        features.build_features(new_rows().drop(columns=["sex"]))


def test_validate_input_reports_problems():
    bad = new_rows().drop(columns=["intake_type"])
    bad.loc[0, "intake_date"] = "yesterday"
    with pytest.raises(predict.InputError, match="missing columns.*invalid intake_date"):
        predict.validate_input(bad, features.INPUT_COLUMNS)


needs_model = pytest.mark.skipif(not config.MODEL_FILE.exists(), reason="run python -m src.train first")


@needs_model
def test_saved_pipeline_scores_unseen_rows_including_unknown_categories():
    bundle = predict.load_bundle()
    out = predict.predict_frame(new_rows(), bundle)
    assert out["risk_probability"].between(0, 1).all()
    assert set(out["high_risk_flag"]) <= {0, 1}
    assert out.loc[1, "risk_probability"] > out.loc[0, "risk_probability"]   # ill severe > normal


@needs_model
def test_cli_writes_output(tmp_path):
    src = tmp_path / "rows.csv"
    new_rows().to_csv(src, index=False)
    assert predict.main([str(src)]) == 0
    assert (tmp_path / "rows_predictions.csv").exists()


@needs_model
def test_cli_rejects_bad_file(tmp_path):
    src = tmp_path / "bad.csv"
    new_rows().drop(columns=["sex"]).to_csv(src, index=False)
    assert predict.main([str(src)]) == 2
