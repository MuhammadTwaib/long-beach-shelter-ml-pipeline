# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Feature definitions shared by training (train.py) and inference (predict.py).

Only information that is known **at the moment of intake** is used. Everything
here is a row-by-row calculation (no averages over the dataset), so it can be
done before the train/test split without leaking test information. All
*fitted* steps (imputation, scaling, encoding) live inside the sklearn Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.cleaning import haversine_km

TARGET = "bad_outcome"

# Columns a new record must contain (the "input contract" checked by predict.py).
INPUT_COLUMNS = [
    "animal_type", "primary_color", "secondary_color", "sex", "dob", "intake_date",
    "intake_condition", "intake_type", "intake_subtype", "reason_for_intake",
    "jurisdiction", "latitude", "longitude",
]

# Columns that must NEVER be features - they are recorded at or after the outcome.
LEAKY_COLUMNS = [
    "outcome_type", "outcome_subtype", "outcome_date", "length_of_stay_days",
    "outcome_is_dead", "was_outcome_alive",
    # Names: the '*' shelter name is usually given when an animal is put up for
    # adoption, i.e. AFTER intake. Named animals have a 7% bad-outcome rate vs 42%
    # for unnamed ones - too good to be true at intake time. Excluded (see leak check).
    "animal_name", "has_name", "name_given_by_shelter",
    # Neutered/spayed: the shelter sterilises animals before adoption and then
    # updates the sex field. 50% of 'spayed' cats/dogs are adopted vs 10% of
    # 'female' ones, so the altered status mostly reflects the outcome. We keep
    # only male / female / unknown (see leak check in train.py).
    "is_altered",
]

NUMERIC_FEATURES = ["age_at_intake_years", "distance_km", "intake_month_sin", "intake_month_cos"]
BINARY_FEATURES = ["has_secondary_color", "reason_given", "age_known"]
CATEGORICAL_FEATURES = [
    "animal_type", "sex_simple", "intake_condition", "intake_type", "intake_subtype",
    "jurisdiction", "primary_color", "intake_dayofweek",
]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def build_features(df):
    """Turn cleaned intake records into model features (row-wise only).

    Accepts text or datetime dates. Returns a new DataFrame with ``FEATURES``.
    """
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Input is missing required columns: {missing}")
    out = pd.DataFrame(index=df.index)
    intake = pd.to_datetime(df["intake_date"], errors="coerce")
    dob = pd.to_datetime(df["dob"], errors="coerce")

    age = (intake - dob).dt.days / 365.25
    age = age.where(age.between(0, config.MAX_AGE_YEARS))          # impossible ages -> NaN
    out["age_at_intake_years"] = age
    out["age_known"] = age.notna().astype(int)
    out["distance_km"] = haversine_km(pd.to_numeric(df["latitude"], errors="coerce").to_numpy(float),
                                      pd.to_numeric(df["longitude"], errors="coerce").to_numpy(float),
                                      config.SHELTER_LAT, config.SHELTER_LON)
    # Month as a point on a circle so December (12) sits next to January (1).
    month = intake.dt.month.to_numpy(dtype=float)
    out["intake_month_sin"] = np.sin(2 * np.pi * month / 12)
    out["intake_month_cos"] = np.cos(2 * np.pi * month / 12)
    out["intake_dayofweek"] = intake.dt.day_name()

    sex = df["sex"].astype("string").str.lower()
    # 'neutered' -> 'male', 'spayed' -> 'female': keep biological sex, drop the altered status.
    out["sex_simple"] = sex.replace({"neutered": "male", "spayed": "female"})
    out["has_secondary_color"] = df["secondary_color"].notna().astype(int)
    out["reason_given"] = df["reason_for_intake"].notna().astype(int)
    for col in ["animal_type", "intake_condition", "intake_type", "intake_subtype",
                "jurisdiction", "primary_color"]:
        out[col] = df[col].astype("string").str.lower().str.strip()
    # sklearn encoders want plain object columns with np.nan, not pandas <NA>.
    cat_cols = CATEGORICAL_FEATURES
    out[cat_cols] = out[cat_cols].astype(object).where(out[cat_cols].notna(), np.nan)
    return out[FEATURES]


def add_leaky_columns(frame, extra):
    """LEAK CHECK ONLY (never used by the saved model): normal features + suspicious columns.

    ``extra`` is a list chosen from ``has_name`` and ``is_altered``.
    """
    out = build_features(frame)
    if "has_name" in extra:
        out["has_name"] = frame["has_name"].astype(int).to_numpy()
    if "is_altered" in extra:
        out["is_altered"] = frame["sex"].astype("string").str.lower().isin(["neutered", "spayed"]).astype(int)
    return out


def make_preprocessor(scale=True):
    """ColumnTransformer: impute + (scale) numerics, pass binaries, one-hot categoricals.

    ``min_frequency=20`` groups rare categories (e.g. 80 colours) into one
    "infrequent" column; unseen categories at predict time are ignored safely.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=False))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
