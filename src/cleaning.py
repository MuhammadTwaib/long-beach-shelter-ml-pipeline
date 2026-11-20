# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Cleaning functions for the Long Beach animal shelter data.

Every function takes a DataFrame (or Series) and returns a NEW object; nothing
is changed in place, and the raw file on disk is never touched.
The order the pipeline calls them in is defined in ``clean_all``.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# Raw header -> snake_case header (also removes the hidden BOM character).
COLUMN_MAP = {
    "Animal ID": "animal_id",
    "Animal Name": "animal_name",
    "Animal Type": "animal_type",
    "Primary Color": "primary_color",
    "Secondary Color": "secondary_color",
    "Sex": "sex",
    "DOB": "dob",
    "Intake Date": "intake_date",
    "Intake Condition": "intake_condition",
    "Intake Type": "intake_type",
    "Intake Subtype": "intake_subtype",
    "Reason for Intake": "reason_for_intake",
    "Outcome Date": "outcome_date",
    "Crossing": "crossing",
    "Jurisdiction": "jurisdiction",
    "Outcome Type": "outcome_type",
    "Outcome Subtype": "outcome_subtype",
    "latitude": "latitude",
    "longitude": "longitude",
    "intake_is_dead": "intake_is_dead",
    "outcome_is_dead": "outcome_is_dead",
    "was_outcome_alive": "was_outcome_alive",
    "geopoint": "geopoint",
}

CATEGORY_COLUMNS = [
    "animal_type", "primary_color", "secondary_color", "sex",
    "intake_condition", "intake_type", "intake_subtype", "reason_for_intake",
    "jurisdiction", "outcome_type", "outcome_subtype",
]

# Spelling mistakes found while inspecting the raw values.
CATEGORY_FIXES = {
    "intake_condition": {"ill moderatete": "ill moderate", "intakeexam": "intake exam"},
    "intake_type": {"euthenasia required": "euthanasia required"},
}

# Colour abbreviations used by the shelter software -> full words.
COLOR_WORDS = {
    "brn": "brown", "br": "brown", "org": "orange", "crm": "cream",
    "slvr": "silver", "bl": "blue", "blk": "black", "choc": "chocolate",
    "pt": "point", "dil": "dilute", "tab": "tabby", "lc": "lilac",
}


def load_raw(path=config.RAW_FILE):
    """Read the raw CSV exactly as published, with every column as text.

    Reading as text first means no value is silently converted; type fixes
    happen later in ``fix_types`` where they are counted.
    ``utf-8-sig`` strips the byte-order mark the portal adds to the header.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run: python -m src.download_data"
        )
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def rename_columns(df):
    """Rename raw headers to snake_case using ``COLUMN_MAP``.

    Raises ``KeyError`` if an expected column is missing, so a changed export
    format fails loudly instead of producing half-empty output.
    """
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise KeyError(f"Raw file is missing expected columns: {sorted(missing)}")
    return df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]


def blank_to_nan(df):
    """Turn empty / whitespace-only strings into real missing values (NaN)."""
    return df.replace(r"^\s*$", np.nan, regex=True)


def standardize_text(series):
    """Lower-case, trim, and collapse repeated spaces in a text column.

    Example: ``"INJURED  SEVERE "`` -> ``"injured severe"``. Missing stays missing.
    """
    return (
        series.str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def standardize_color(series):
    """Expand shelter colour abbreviations word by word.

    Example: ``"brn tabby"`` -> ``"brown tabby"``, ``"seal pt"`` -> ``"seal point"``.
    """
    def expand(value):
        if pd.isna(value):
            return value
        return " ".join(COLOR_WORDS.get(word, word) for word in value.split())

    return standardize_text(series).map(expand)


def standardize_categories(df):
    """Apply text standardisation, colour expansion and known spelling fixes.

    Also merges the nine city council districts (``district1`` ... ``district9``)
    into ``long beach`` because they are parts of the same city.
    """
    out = df.copy()
    for col in CATEGORY_COLUMNS:
        out[col] = standardize_text(out[col])
    for col in ("primary_color", "secondary_color"):
        out[col] = standardize_color(out[col])
    for col, fixes in CATEGORY_FIXES.items():
        out[col] = out[col].replace(fixes)
    out["jurisdiction"] = out["jurisdiction"].replace(r"^district\d$", "long beach", regex=True)
    return out


def clean_names(df):
    """Clean the animal name and add two flags.

    The shelter prefixes names it invented with ``*`` (e.g. ``*CHARLIEN``).
    ``has_name``  - the animal has any name.
    ``name_given_by_shelter`` - the name started with ``*``.
    A name that is only ``*`` counts as no name.
    """
    out = df.copy()
    name = out["animal_name"].str.strip()
    out["name_given_by_shelter"] = name.str.startswith("*", na=False)
    name = name.str.lstrip("*").str.strip().str.title()
    out["animal_name"] = name.replace("", np.nan)
    out["has_name"] = out["animal_name"].notna()
    return out


def clean_address(series):
    """Remove stray leading punctuation (``[``, `` ` ``, quotes) from addresses."""
    return series.str.replace(r"^[^0-9A-Za-z]+", "", regex=True).str.strip()


def fix_types(df):
    """Convert text columns to proper dtypes.

    Dates -> ``datetime64`` (bad strings become NaT), coordinates -> float,
    the two outcome flags -> nullable boolean / integer.
    Returns the new frame and a dict counting values that failed to parse.
    """
    out = df.copy()
    failures = {}
    for col in ("dob", "intake_date", "outcome_date"):
        parsed = pd.to_datetime(out[col], format="%Y-%m-%d", errors="coerce")
        failures[col] = int((parsed.isna() & out[col].notna()).sum())
        out[col] = parsed
    for col in ("latitude", "longitude"):
        parsed = pd.to_numeric(out[col], errors="coerce")
        failures[col] = int((parsed.isna() & out[col].notna()).sum())
        out[col] = parsed
    out["outcome_is_dead"] = out["outcome_is_dead"].map({"TRUE": True, "FALSE": False}).astype("boolean")
    out["was_outcome_alive"] = pd.to_numeric(out["was_outcome_alive"], errors="coerce").astype("Int64")
    out["crossing"] = clean_address(out["crossing"])
    return out, failures


def drop_redundant_columns(df):
    """Drop columns that carry no extra information.

    ``geopoint`` repeats latitude/longitude; ``intake_is_dead`` has a single value
    (``Alive on Intake``) for every row.
    """
    return df.drop(columns=["geopoint", "intake_is_dead"])


def remove_duplicates(df):
    """Remove duplicate records and return (kept, removed).

    Two rules, decided after inspecting the data:
    1. Rows whose outcome is literally ``duplicate`` are system duplicates.
    2. Rows with the same animal_id AND intake_date describe the same stay;
       we keep the last one (the most recently updated outcome).
    The same animal on *different* dates is a real repeat intake and is kept.
    """
    is_dup_outcome = df["outcome_type"].eq("duplicate")
    is_repeat_key = df.duplicated(subset=["animal_id", "intake_date"], keep="last")
    removed_mask = is_dup_outcome | is_repeat_key
    return df[~removed_mask].copy(), df[removed_mask].copy()


def fix_invalid_values(df):
    """Blank out values that are impossible, keeping the rest of the row.

    * DOB after the intake date, or an age above ``MAX_AGE_YEARS`` -> DOB = NaT
    * coordinates outside Southern California (geocoding errors) -> both set to NaN
    Returns the new frame and a dict with how many values were blanked.
    """
    out = df.copy()
    age_days = (out["intake_date"] - out["dob"]).dt.days
    bad_dob = (age_days < 0) | (age_days > config.MAX_AGE_YEARS * 365.25)
    out.loc[bad_dob, "dob"] = pd.NaT

    lat_ok = out["latitude"].between(*config.LAT_RANGE)
    lon_ok = out["longitude"].between(*config.LON_RANGE)
    bad_geo = ~(lat_ok & lon_ok) & out["latitude"].notna()
    out.loc[bad_geo, ["latitude", "longitude"]] = np.nan
    return out, {"dob_invalid_blanked": int(bad_dob.sum()), "coords_outside_socal_blanked": int(bad_geo.sum())}


def add_derived_columns(df):
    """Add analysis columns using vectorised NumPy operations.

    * ``age_at_intake_years`` - (intake - dob) in years
    * ``length_of_stay_days`` - (outcome - intake) in days, clipped at ``MAX_STAY_DAYS``
    * ``age_group``  - ``np.select`` buckets: baby / young / adult / senior / unknown
    * ``distance_km`` - haversine distance from the shelter (NumPy trig on arrays)
    * ``bad_outcome`` - 1 if the animal died/was euthanised, 0 if it left alive,
      missing if the outcome is unknown, missing or a duplicate marker.
    """
    out = df.copy()
    out["age_at_intake_years"] = ((out["intake_date"] - out["dob"]).dt.days / 365.25).round(2)
    stay = (out["outcome_date"] - out["intake_date"]).dt.days
    out["length_of_stay_days"] = np.clip(stay, 0, config.MAX_STAY_DAYS)

    age = out["age_at_intake_years"].to_numpy(dtype=float)
    out["age_group"] = np.select(
        [age < 0.5, age < 2, age < 8, age >= 8],
        ["baby (<6m)", "young (6m-2y)", "adult (2-8y)", "senior (8y+)"],
        default="unknown",
    )
    out["distance_km"] = haversine_km(out["latitude"].to_numpy(dtype=float),
                                      out["longitude"].to_numpy(dtype=float),
                                      config.SHELTER_LAT, config.SHELTER_LON).round(2)

    outcome = out["outcome_type"]
    label = np.where(outcome.isin(config.BAD_OUTCOMES), 1.0, 0.0)
    unknown = outcome.isna() | outcome.isin(config.NON_OUTCOMES)
    out["bad_outcome"] = pd.Series(np.where(unknown, np.nan, label), index=out.index).astype("Int64")
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between arrays of points (fully vectorised).

    NaN coordinates give NaN distance instead of an error.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def clean_all(raw):
    """Run every cleaning step in order.

    Returns ``(clean_df, rejected_df, log)`` where ``log`` holds counts used in
    the data-quality report.
    """
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("clean_all expects a non-empty DataFrame")
    log = {"rows_raw": len(raw)}
    df = rename_columns(raw)
    df = blank_to_nan(df)
    df = standardize_categories(df)
    df = clean_names(df)
    df, log["parse_failures"] = fix_types(df)
    df = drop_redundant_columns(df)
    df, rejected = remove_duplicates(df)
    log["duplicates_removed"] = len(rejected)
    df, blanked = fix_invalid_values(df)
    log.update(blanked)
    df = add_derived_columns(df)
    log["rows_clean"] = len(df)
    return df.reset_index(drop=True), rejected.reset_index(drop=True), log


# Pattern exported for tests / validation of IDs.
ANIMAL_ID_PATTERN = re.compile(r"^A\d{6}$")
