# Assumptions, limitations, risks and model card

## Part A - data-cleaning decisions

| # | Decision | Why | Effect |
|---|---|---|---|
| 1 | Read every column as **text** first, then parse types | No silent conversion; parse failures are counted | 0 parse failures on the real file |
| 2 | Lower-case, trim and collapse double spaces in categories | `INJURED  SEVERE` and `injured severe` must be one value | Intake condition 17 → 16 values |
| 3 | Fix typos `ill moderatete`, `euthenasia required`, `intakeexam` | Found by listing all unique values | — |
| 4 | Expand colour codes (`brn`→`brown`, `pt`→`point`…) | Same colour was spelled several ways | Colours easier to read; same number of rows |
| 5 | Merge `district1…district9` into `long beach` | Council districts are inside Long Beach city | Jurisdiction 26 → 16 values |
| 6 | Remove rows with outcome `DUPLICATE` and repeated (animal_id, intake_date) - **keep the last** | Both describe the same stay twice; the last row is the latest update | **50 rows removed**, saved to `shelter_rejected.csv` |
| 7 | **Keep** the same animal on different dates | These are real repeat intakes (1,033 IDs repeat) | — |
| 8 | DOB after intake date or age > 30 y → set DOB to missing | Impossible; probably typing errors (max DOB was 2031) | 149 values blanked |
| 9 | Coordinates outside Southern California → missing | Hawaii, Oregon, New York points are geocoding errors. Points in Tustin/Riverside etc. are **kept** - the shelter does take some out-of-area animals | 8 rows blanked |
| 10 | Drop `geopoint` and `intake_is_dead` | Duplicate of lat/lon; constant column | 23 → 21 raw columns kept |
| 11 | Strip leading `[` and `` ` `` from addresses | Data-entry noise | — |
| 12 | Length of stay clipped to 0-1,500 days | Guard for future bad dates; no real value was clipped (max 1,107) | — |
| 13 | Target `bad_outcome`: 1 = euthanasia / died / **disposal**; 0 = any live outcome; missing = blank or `MISSING` outcome | The raw `was_outcome_alive` marks `DISPOSAL` (disposal of a dead animal) as *alive* - we think that is an error | 245 rows have no usable label (kept in Part A, excluded from Part B) |

### What I chose **not** to fix (and why)
* **Missing `reason_for_intake` (93%)** - it is only asked when an owner surrenders an animal. Filling it would invent data; instead Part B uses a flag `reason_given`.
* **Missing `animal_name` (41%) and `secondary_color` (52%)** - these are real "no name" / "one colour", not errors.
* **Missing DOB (12% + 149 blanked)** - not imputed in Part A. In Part B the median is imputed *inside* the pipeline (fitted on training rows only) and `age_known` tells the model it was missing.
* **Addresses (`crossing`)** - very inconsistent free text; not standardised because coordinates already give location.
* **Outcome subtype (223 codes)** - left as-is; only used for error analysis, never as a feature.
* **Long stays (up to 1,107 days)** - kept; they are rare but possible (long-term holds, court cases).

### Assumptions
* The CSV published on the City of Long Beach portal (mirrored on GitHub, pinned commit + SHA-256) is the full record for 2017-2024.
* Dates are local dates without time; "same day" stays have length 0.
* Validation before cleaning shows **0% row pass rate** mainly because raw categories are UPPER CASE and not in the allowed lists - this is expected; the point of the metric is that cleaning brings it to 100%.

### Limitations
* Mean missing % rises slightly after cleaning (10.12 → 10.15) because we **turned impossible values into missing** on purpose.
* The 2020-2021 COVID period has far fewer intakes; summaries that mix years are affected.
* Shelter-specific codes (`OTC`, `ACS DISPO`) are interpreted from context, not from an official code book.

---

## Part B - Model card: "Intake risk score"

**Model:** `HistGradientBoostingClassifier` (class_weight=balanced, learning_rate 0.05, max_leaf_nodes 31, l2 1.0) inside one sklearn `Pipeline` (feature builder → ColumnTransformer → model). File: `models/outcome_pipeline.joblib` (+ decision threshold 0.657).

**What it predicts:** probability that an animal, *at the moment of intake*, will **not leave the shelter alive** (euthanised, died, disposal). Positive class ≈ 20% of intakes.

**Intended use:** a triage aid for intake staff - put high-scoring animals first in the queue for a vet check, medical treatment or a call to rescue partners.
**Not intended for:** deciding euthanasia, rejecting animals, judging owners or neighbourhoods, or any use outside Long Beach Animal Care Services without re-validation.

**Training data:** 21,302 intakes 2017-2022 (train), 3,976 from 2023 (validation, used to choose model and threshold), refit on 2017-2023 for the final model.
**Evaluation data:** 4,109 intakes 1 Jan - 14 Dec 2024, used **once**. A further 105 intakes (15-31 Dec 2024) were never used and are scored by `predict.py`.

**Why these metrics:** the classes are imbalanced (80/20), so accuracy would reward a model that says "alive" for everyone (80% accurate, useless). We report **PR-AUC** (main metric - how well the risky animals are ranked at the top), **recall** (share of at-risk animals we catch - missing one is the costly error), **precision** (how many flags are real - each flag costs staff time) and F1. Threshold = the one that maximised F1 on 2023.

**Results (2024 test):** PR-AUC **0.72** (dummy 0.20), ROC-AUC 0.88, recall 0.72, precision 0.61, F1 0.66. On the 105 unseen rows: PR-AUC 0.75, recall 0.87, precision 0.74.

**Features used:** animal type, sex (male/female/unknown only), age at intake, intake condition / type / subtype, whether a surrender reason was given, jurisdiction, primary colour, whether there is a secondary colour, distance from the shelter, month (as sine/cosine), weekday.
**Excluded because they leak the answer:** outcome type/subtype/date, length of stay, `outcome_is_dead`, `was_outcome_alive`; **animal name** (names marked `*` are given when an animal is put up for adoption - adding `has_name` raised validation PR-AUC from 0.70 to 0.78, too good to be true at intake) and **neutered/spayed status** (animals are sterilised before adoption and the sex field is then updated; 50% of "spayed" cats/dogs are adopted vs 10% of "female" ones; adding it raised PR-AUC to 0.76).

**Most important features (permutation importance, 2024):** intake condition (by far), sex, animal type, date of birth (age), primary colour. This makes sense: severely ill or injured animals and wildlife are the ones most often euthanised.

**Known failure patterns (from `outputs/reports/error_examples.csv`, `subgroup_performance.csv`):**
1. **Healthy-looking dogs later euthanised for behaviour** - all the most confident misses are dogs with intake condition `normal` whose outcome subtype is `beh severe` or `at vet`. Behaviour problems are not visible at intake; recall for dogs is only **0.31**.
2. **Animals that get sick in the shelter** - the largest group of misses has outcome subtype `ill severe` although they arrived `normal`/`under age/weight`. Intake data cannot see the future illness.
3. **Over-flagging severely ill wildlife and "other" animals that were transferred or rescued** - e.g. `ill severe` wildlife with outcome `transfer`; and sick community cats returned to the field. Recall for wildlife is 1.0 but precision 0.58.
4. Groups with very few positives (`return`, `welfare seized`, intake condition `normal`) have unstable metrics.

**Risks & ethics:**
* *Self-fulfilling prophecy* - if staff give less effort to high scores, those animals will do worse and the model will look "right". The score must mean "help first", and outcomes should be monitored by score band.
* `sex = unknown` is still an important feature and may partly reflect how quickly an animal was processed - a remaining possible leak to check with the shelter.
* Policy changes (e.g. a new no-kill target) change the label itself; the model must be re-validated after such changes and retrained regularly.
* Coordinates locate where an animal was found; the model must not be used to rate neighbourhoods or owners.
