## What does this PR change?
<!-- One or two sentences. Link the checkpoint / issue. -->

## Why?

## How was it tested?
- [ ] `pytest -q` passes locally
- [ ] `ruff check src tests` passes
- [ ] Pipeline re-run (`python -m src.pipeline_a` / `python -m src.train`) if data or model code changed
- [ ] New/changed functions have a docstring and at least one test

## Data & model safety
- [ ] Raw data in `data/raw/` is untouched
- [ ] No new feature is recorded after intake (leakage check)
- [ ] The test set was not used for any decision
- [ ] Outputs/figures regenerated and committed if they changed

## Screenshots / numbers (if outputs changed)
<!-- before vs after metrics, a figure, etc. -->
