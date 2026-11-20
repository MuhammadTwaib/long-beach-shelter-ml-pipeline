# Raw data schema

**File:** `data/raw/animal-shelter-intakes-and-outcomes.csv` (never edited)
**Rows × columns:** 29,787 × 23 · **Encoding:** UTF-8 with a byte-order mark (BOM) in the header
**Grain:** one row = one stay (intake → outcome) of one animal. The same `Animal ID` can appear again if the animal comes back (1,033 repeat IDs).
**Period:** intake dates 2017-01-01 to 2024-12-31.

"Missing" below counts empty strings in the raw file. Every column is read as text first; types are fixed in `src/cleaning.py::fix_types`.

| # | Raw column | Clean name | Raw type → clean type | Unit / format | Allowed values / rule | Missing (raw) | Unique | Problems found |
|---|---|---|---|---|---|---|---|---|
| 1 | Animal ID | animal_id | text → text | `A` + 6 digits | regex `^A\d{6}$` | 0 | 28,754 | none in format; 43 rows share ID **and** intake date (duplicates) |
| 2 | Animal Name | animal_name | text → text | — | free text | 12,162 (40.8%) | 9,206 | `*` prefix = name invented by shelter; 1 name is just `*`; ALL CAPS |
| 3 | Animal Type | animal_type | text → category | — | cat, dog, bird, wild, other, rabbit, reptile, guinea pig, livestock, amphibian | 0 | 10 | ALL CAPS |
| 4 | Primary Color | primary_color | text → category | — | 80 values | 0 | 80 | abbreviations (`BRN TABBY`, `SEAL PT`, `ORG TABBY`) |
| 5 | Secondary Color | secondary_color | text → category | — | 50 values | 15,604 (52.4%) | 50 | same abbreviations; missing = single-coloured |
| 6 | Sex | sex | text → category | — | male, female, neutered, spayed, unknown | 0 | 5 | mixes sex and neuter status in one field |
| 7 | DOB | dob | text → date | YYYY-MM-DD | ≤ intake date, age ≤ 30 y | 3,591 (12.1%) | 5,656 | 149 DOBs **after** intake date (max 2031) |
| 8 | Intake Date | intake_date | text → date | YYYY-MM-DD | 2017–2024 | 0 | 2,828 | none |
| 9 | Intake Condition | intake_condition | text → category | — | 16 values after fixes | 0 | 17 | double spaces (`INJURED  SEVERE`), typo `ILL MODERATETE`, `INTAKEEXAM` |
| 10 | Intake Type | intake_type | text → category | — | 12 values | 0 | 12 | mixed case + typo `Euthenasia Required` |
| 11 | Intake Subtype | intake_subtype | text → category | — | 23 codes | 390 (1.3%) | 23 | short codes (`OTC` = over the counter) |
| 12 | Reason for Intake | reason_for_intake | text → category | — | 58 codes | 27,784 (93.3%) | 58 | only filled for owner surrenders |
| 13 | Outcome Date | outcome_date | text → date | YYYY-MM-DD | ≥ intake date | 177 (0.6%) | 2,832 | missing = still in shelter / not recorded |
| 14 | Crossing | crossing | text → text | street address | — | 0 | 15,692 | stray `[` and `` ` `` at the start; inconsistent formats |
| 15 | Jurisdiction | jurisdiction | text → category | — | city names | 1 | 26 | `DISTRICT1`…`DISTRICT9` are parts of Long Beach |
| 16 | Outcome Type | outcome_type | text → category | — | 18 values | 187 (0.6%) | 19 | `DUPLICATE` and `MISSING` are not real outcomes |
| 17 | Outcome Subtype | outcome_subtype | text → category | — | 223 codes | 3,386 (11.4%) | 223 | very high cardinality; partly repeats outcome |
| 18 | latitude | latitude | text → float | decimal degrees | 32.5–35.0 (S. California) | 0 | 9,075 | 8 points far away (Hawaii, Oregon, New York) |
| 19 | longitude | longitude | text → float | decimal degrees | −119.5 to −116.0 | 0 | 8,972 | see latitude |
| 20 | intake_is_dead | *(dropped)* | text | — | — | 0 | 1 | constant value `Alive on Intake` → no information |
| 21 | outcome_is_dead | outcome_is_dead | text → boolean | TRUE/FALSE | — | 0 | 2 | **leaks the target** |
| 22 | was_outcome_alive | was_outcome_alive | text → int | 0/1 | — | 0 | 2 | **leaks the target**; marks `DISPOSAL` (dead-animal disposal) as alive |
| 23 | geopoint | *(dropped)* | text | "lat, lon" | — | 0 | 9,273 | repeats latitude + longitude |

## Derived columns added by the pipeline

| Column | Type | Unit | How |
|---|---|---|---|
| has_name, name_given_by_shelter | bool | — | from `animal_name` |
| age_at_intake_years | float | years | (intake_date − dob) / 365.25 |
| length_of_stay_days | float | days | outcome_date − intake_date, clipped to 0–1500 |
| age_group | category | — | `np.select` buckets: baby <6 m, young 6 m–2 y, adult 2–8 y, senior 8 y+, unknown |
| distance_km | float | km | haversine distance from the shelter (7700 E Spring St) |
| bad_outcome | int (0/1/NA) | — | **Part B target.** 1 = euthanasia / died / disposal; 0 = any live outcome; NA = no, missing or `DUPLICATE` outcome |
