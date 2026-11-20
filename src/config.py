# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Central settings: file paths, constants and thresholds used by every script.

Keeping them in one place means a reviewer can change, say, the quality-gate
threshold without hunting through the code.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]          # project/ folder
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "cleaned"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_DIR = ROOT / "models"
SQL_DIR = ROOT / "sql"

RAW_FILE = RAW_DIR / "animal-shelter-intakes-and-outcomes.csv"
CLEAN_FILE = CLEAN_DIR / "shelter_clean.csv"
REJECTED_FILE = CLEAN_DIR / "shelter_rejected.csv"
DB_FILE = CLEAN_DIR / "shelter.db"
METRICS_FILE = REPORT_DIR / "metrics.json"
MODEL_FILE = MODEL_DIR / "outcome_pipeline.joblib"

# ---------------------------------------------------------------- data source
RAW_URL = (
    "https://raw.githubusercontent.com/EmilHvitfeldt/animalshelter/"
    "1bd593f82b2384d70d3b38aa44f9c4606bb3fccc/data-raw/"
    "animal-shelter-intakes-and-outcomes.csv"
)
RAW_SHA256 = "3228dd3a6d08033dba1a41815e7dadc22298aeb4117a240f36979ff5b2e528f1"

# ---------------------------------------------------------------- domain constants
# Long Beach Animal Care Services, 7700 E Spring St (approximate coordinates).
SHELTER_LAT, SHELTER_LON = 33.8105, -118.1225

# Box around Southern California. Some intakes really come from outside Long
# Beach (Tustin, Riverside...), so only points outside this much wider box
# (e.g. Hawaii, New York) are treated as geocoding errors.
LAT_RANGE = (32.5, 35.0)
LON_RANGE = (-119.5, -116.0)

MAX_AGE_YEARS = 30          # oldest plausible shelter animal
MAX_STAY_DAYS = 1500        # longest plausible stay; the raw max is ~1,100

# Outcomes that mean the animal did NOT leave alive -> target = 1 in Part B.
BAD_OUTCOMES = {"euthanasia", "died", "disposal"}
# Outcome values that are not real outcomes -> no usable label.
NON_OUTCOMES = {"duplicate", "missing"}

# ---------------------------------------------------------------- quality gate
VALIDATION_PASS_THRESHOLD = 0.95    # pipeline stops if fewer checks pass

RANDOM_STATE = 42
TEST_START_DATE = "2024-01-01"      # Part B: intakes from 2024 form the test set
