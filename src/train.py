# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Part B: frame the problem, split by time, compare models, analyse errors, save the pipeline.

Problem: at intake, predict whether an animal will NOT leave the shelter alive
(died / euthanised) -> binary classification, positive class = 1 (~21%).

Split (time-based, made BEFORE any fitting):
    train      intakes 2017-01-01 .. 2022-12-31   fit + 5-fold time-series CV
    validation intakes 2023                       model choice + decision threshold
    test       intakes 2024-01-01 .. 2024-12-14   used ONCE at the end
    unseen     intakes 2024-12-15 .. 2024-12-31   never touched; written to data/new/ for predict.py

Run:  python -m src.train
"""

import json
import sys
from functools import partial

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402

from src import config, features  # noqa: E402

VAL_START, TEST_START, UNSEEN_START = "2023-01-01", config.TEST_START_DATE, "2024-12-15"
NEW_DIR = config.ROOT / "data" / "new"
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1",
           "recall": "recall", "precision": make_scorer(precision_score, zero_division=0)}
BLUE, ORANGE, GREEN, VIOLET, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#52514e"


# ------------------------------------------------------------------ data
def load_labelled(path=config.CLEAN_FILE):
    """Load the Part A output and keep rows with a known outcome, oldest first."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run Part A first: python -m src.pipeline_a")
    df = pd.read_csv(path, parse_dates=["dob", "intake_date", "outcome_date"])
    df = df[df[features.TARGET].notna()].copy()
    df[features.TARGET] = df[features.TARGET].astype(int)
    return df.sort_values("intake_date", kind="stable").reset_index(drop=True)


def time_split(df):
    """Split by intake date into train / validation / test / unseen (no overlap)."""
    d = df["intake_date"]
    parts = {
        "train": df[d < VAL_START],
        "validation": df[(d >= VAL_START) & (d < TEST_START)],
        "test": df[(d >= TEST_START) & (d < UNSEEN_START)],
        "unseen": df[d >= UNSEEN_START],
    }
    assert sum(len(p) for p in parts.values()) == len(df), "split lost rows"
    return parts


# ------------------------------------------------------------------ models
def make_pipeline(model, scale=True, feature_fn=features.build_features):
    """One object: feature building -> preprocessing -> model."""
    return Pipeline([
        ("features", FunctionTransformer(feature_fn)),
        ("prep", features.make_preprocessor(scale=scale)),
        ("model", model),
    ])


def candidate_models():
    """Baseline + three candidates. class_weight='balanced' because only ~21% are positive."""
    rs = config.RANDOM_STATE
    return {
        "Baseline: Dummy (prior)": make_pipeline(DummyClassifier(strategy="prior")),
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                   n_jobs=-1, random_state=rs), scale=False),
        "Hist Gradient Boosting": make_pipeline(
            HistGradientBoostingClassifier(class_weight="balanced", random_state=rs), scale=False),
    }


def scores(y_true, proba, threshold=0.5):
    """All metrics for one set of predictions."""
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
    }


def best_f1_threshold(y_true, proba):
    """Threshold that maximises F1 on the given (validation) data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


# ------------------------------------------------------------------ plots
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_pr_curves(curves, base_rate, path):
    """Precision-recall curves of all models on the validation year."""
    fig, ax = plt.subplots(figsize=(7, 5))
    # The dummy gives one constant score, so its "curve" is just the dotted base-rate line.
    real = {k: v for k, v in curves.items() if not k.startswith("Baseline")}
    for (name, (y, p)), color in zip(real.items(), [BLUE, GREEN, ORANGE, VIOLET]):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{name} (PR-AUC {average_precision_score(y, p):.2f})")
    ax.axhline(base_rate, color=MUTED, linestyle=":", linewidth=1)
    ax.text(0.01, base_rate + 0.015, f"Baseline: Dummy = base rate (PR-AUC {base_rate:.2f})", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax, "Precision-recall on validation year 2023", "Recall (share of bad outcomes caught)",
           "Precision (share of flags that were right)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y, pred, path):
    """Confusion matrix on the 2024 test set."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=["left alive", "died/euth."]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=",")
    ax.set_title("Confusion matrix, test set 2024 (counts of animals)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp, path):
    """Horizontal bars of permutation importance (drop in PR-AUC)."""
    imp = imp.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"], color=BLUE,
            error_kw={"ecolor": MUTED, "elinewidth": 1})
    _style(ax, "Permutation importance on the 2024 test set", "Drop in PR-AUC when the feature is shuffled",
           "Feature")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ error analysis
def subgroup_table(frame, proba, threshold, column):
    """Metrics per subgroup (only groups with >= 30 rows and both classes)."""
    rows = []
    for value, idx in frame.groupby(column).groups.items():
        y, p = frame.loc[idx, features.TARGET], proba[frame.index.get_indexer(idx)]
        if len(y) < 30 or y.nunique() < 2:
            continue
        s = scores(y, p, threshold)
        rows.append({"group_by": column, "group": value, "n": len(y), "positive_rate": round(y.mean(), 3),
                     **{k: round(v, 3) for k, v in s.items()}})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    """Train, compare, evaluate, explain and save. Writes everything to outputs/ and models/."""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    NEW_DIR.mkdir(parents=True, exist_ok=True)

    df = load_labelled()
    parts = time_split(df)
    for name, part in parts.items():
        print(f"{name:<10} {len(part):>6} rows  {part['intake_date'].min().date()} .. "
              f"{part['intake_date'].max().date()}  positive rate {part[features.TARGET].mean():.3f}")
    X = {k: v.drop(columns=[features.TARGET]) for k, v in parts.items()}
    y = {k: v[features.TARGET] for k, v in parts.items()}

    # The unseen rows are written WITHOUT outcome columns - like real new intakes.
    unseen_cols = ["animal_id"] + features.INPUT_COLUMNS
    parts["unseen"][unseen_cols].to_csv(NEW_DIR / "unseen_intakes.csv", index=False)
    parts["unseen"][["animal_id", features.TARGET]].to_csv(NEW_DIR / "unseen_answers.csv", index=False)

    # ---- 1. cross-validation on train + validation-year scores (same split for every model)
    cv = TimeSeriesSplit(n_splits=5)
    models = candidate_models()
    rows, curves, fitted = [], {}, {}
    for name, pipe in models.items():
        print(f"CV: {name}")
        cvres = cross_validate(pipe, X["train"], y["train"], cv=cv, scoring=SCORING, n_jobs=1)
        pipe.fit(X["train"], y["train"])
        proba = pipe.predict_proba(X["validation"])[:, 1]
        val = scores(y["validation"], proba)
        curves[name] = (y["validation"], proba)
        fitted[name] = pipe
        row = {"model": name}
        for m in SCORING:
            row[f"cv_{m}"] = f"{cvres['test_' + m].mean():.3f} ± {cvres['test_' + m].std():.3f}"
        row.update({f"val_{k}": round(v, 3) for k, v in val.items()})
        rows.append(row)

    # ---- 2. small hyperparameter search on TRAIN only, for the boosting model
    print("Grid search: Hist Gradient Boosting")
    grid = GridSearchCV(
        make_pipeline(HistGradientBoostingClassifier(class_weight="balanced",
                                                     random_state=config.RANDOM_STATE), scale=False),
        {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
         "model__l2_regularization": [0.0, 1.0]},
        scoring="average_precision", cv=cv, n_jobs=1)
    grid.fit(X["train"], y["train"])
    tuned_name = "Hist Gradient Boosting (tuned)"
    proba = grid.best_estimator_.predict_proba(X["validation"])[:, 1]
    curves[tuned_name] = (y["validation"], proba)
    fitted[tuned_name] = grid.best_estimator_
    best_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
    rows.append({"model": tuned_name, "cv_pr_auc": f"{grid.best_score_:.3f} (best of grid)",
                 **{f"val_{k}": round(v, 3) for k, v in scores(y["validation"], proba).items()}})

    # ---- 3. leakage check: what if we (wrongly) used columns recorded after intake?
    for extra in (["has_name"], ["is_altered"]):
        leak_pipe = make_pipeline(
            HistGradientBoostingClassifier(class_weight="balanced", random_state=config.RANDOM_STATE),
            scale=False, feature_fn=partial(features.add_leaky_columns, extra=extra))
        # Tell the preprocessor to pass the extra column straight through as well.
        leak_pipe.named_steps["prep"].transformers[1] = ("bin", "passthrough", features.BINARY_FEATURES + extra)
        leak_pipe.fit(X["train"], y["train"])
        leak_val = scores(y["validation"], leak_pipe.predict_proba(X["validation"])[:, 1])
        rows.append({"model": f"LEAK CHECK - HGB + {extra[0]} (not used)",
                     **{f"val_{k}": round(v, 3) for k, v in leak_val.items()}})

    results = pd.DataFrame(rows)
    results.to_csv(config.REPORT_DIR / "model_comparison.csv", index=False)
    print(results.to_string(index=False))
    plot_pr_curves(curves, y["validation"].mean(), config.FIG_DIR / "06_pr_curves_validation.png")

    # ---- 4. choose the model + threshold on VALIDATION (never on test)
    candidates = results[~results["model"].str.startswith(("LEAK", "Baseline"))]
    chosen = candidates.sort_values("val_pr_auc", ascending=False).iloc[0]["model"]
    threshold = best_f1_threshold(*curves[chosen])
    print(f"Chosen: {chosen}  threshold={threshold:.3f}")

    # ---- 5. refit chosen model on train + validation, evaluate ONCE on test
    final = fitted[chosen]
    X_dev = pd.concat([X["train"], X["validation"]])
    y_dev = pd.concat([y["train"], y["validation"]])
    final.fit(X_dev, y_dev)
    test_proba = final.predict_proba(X["test"])[:, 1]
    test_scores = scores(y["test"], test_proba, threshold)
    dummy_test = scores(y["test"], np.full(len(y["test"]), y_dev.mean()), threshold)
    test_table = pd.DataFrame([
        {"model": "Baseline: Dummy (prior)", **{k: round(v, 3) for k, v in dummy_test.items()}},
        {"model": f"{chosen} (final)", **{k: round(v, 3) for k, v in test_scores.items()}},
    ])
    test_table.to_csv(config.REPORT_DIR / "test_results.csv", index=False)
    print(test_table.to_string(index=False))
    test_pred = (test_proba >= threshold).astype(int)
    plot_confusion(y["test"], test_pred, config.FIG_DIR / "07_confusion_matrix_test.png")

    # ---- 6. error analysis on test
    test_frame = parts["test"].copy()
    test_frame["proba"] = test_proba
    test_frame["pred"] = test_pred
    test_frame["error"] = np.select(
        [(test_pred == 1) & (y["test"] == 0), (test_pred == 0) & (y["test"] == 1)],
        ["false positive", "false negative"], default="correct")
    show = ["animal_id", "animal_type", "age_at_intake_years", "sex", "intake_condition", "intake_type",
            "jurisdiction", "outcome_type", "outcome_subtype", "proba", "error"]
    worst_fn = test_frame[test_frame["error"] == "false negative"].nsmallest(15, "proba")[show]
    worst_fp = test_frame[test_frame["error"] == "false positive"].nlargest(15, "proba")[show]
    pd.concat([worst_fn, worst_fp]).to_csv(config.REPORT_DIR / "error_examples.csv", index=False)
    pd.concat([subgroup_table(test_frame.reset_index(drop=True), test_proba, threshold, col)
               for col in ["animal_type", "intake_type", "age_group", "intake_condition"]]
              ).to_csv(config.REPORT_DIR / "subgroup_performance.csv", index=False)
    fn_outcomes = (test_frame[test_frame["error"] == "false negative"]["outcome_subtype"]
                   .value_counts().head(10))
    fn_outcomes.to_csv(config.REPORT_DIR / "false_negative_outcome_subtypes.csv")

    # ---- 7. permutation importance (on raw input columns -> easy to explain)
    imp = permutation_importance(final, X["test"][features.INPUT_COLUMNS], y["test"],
                                 scoring="average_precision", n_repeats=5,
                                 random_state=config.RANDOM_STATE, n_jobs=1)
    imp_df = pd.DataFrame({"feature": features.INPUT_COLUMNS, "importance_mean": imp.importances_mean,
                           "importance_std": imp.importances_std}).sort_values("importance_mean",
                                                                                ascending=False)
    imp_df.round(4).to_csv(config.REPORT_DIR / "permutation_importance.csv", index=False)
    plot_importance(imp_df, config.FIG_DIR / "08_permutation_importance.png")

    # ---- 8. save the single pipeline object + the metadata predict.py needs
    bundle = {
        "pipeline": final, "threshold": threshold, "model_name": chosen,
        "input_columns": features.INPUT_COLUMNS, "trained_on": f"intakes before {TEST_START}",
        "sklearn_version": sklearn.__version__, "best_params": best_params,
    }
    joblib.dump(bundle, config.MODEL_FILE, compress=3)
    summary = {"chosen_model": chosen, "threshold": round(threshold, 4), "best_grid_params": best_params,
               "test": {k: round(v, 4) for k, v in test_scores.items()},
               "split_sizes": {k: len(v) for k, v in parts.items()}}
    (config.REPORT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved pipeline to {config.MODEL_FILE.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
