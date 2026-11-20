# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Compare predict.py output on the held-back unseen rows with their real outcomes.

    python -m src.predict data/new/unseen_intakes.csv
    python -m src.evaluate_unseen
"""

import json
import sys

import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src import config

NEW_DIR = config.ROOT / "data" / "new"


def main():
    """Join predictions to answers by row order and write a small JSON report."""
    pred_file = NEW_DIR / "unseen_intakes_predictions.csv"
    if not pred_file.exists():
        print("Run first: python -m src.predict data/new/unseen_intakes.csv", file=sys.stderr)
        return 1
    pred = pd.read_csv(pred_file)
    answers = pd.read_csv(NEW_DIR / "unseen_answers.csv")
    if len(pred) != len(answers) or not (pred["animal_id"] == answers["animal_id"]).all():
        print("Predictions and answers are not aligned row by row", file=sys.stderr)
        return 1
    y = answers["bad_outcome"]
    report = {
        "rows": len(y), "positive_rate": round(float(y.mean()), 3),
        "pr_auc": round(average_precision_score(y, pred["risk_probability"]), 3),
        "recall": round(recall_score(y, pred["high_risk_flag"]), 3),
        "precision": round(precision_score(y, pred["high_risk_flag"]), 3),
    }
    (config.REPORT_DIR / "unseen_rows_scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
