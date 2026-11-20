# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Summaries (Pandas/NumPy), charts (Matplotlib) and the SQLite query for Part A.

Everything is written to ``outputs/``; nothing depends on notebook state.
"""

import sqlite3

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed (works on servers / CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def _style(ax, title, xlabel, ylabel):
    """Apply one consistent, quiet style: title, labelled axes, light grid."""
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


# ------------------------------------------------------------------ summaries
def summary_tables(df):
    """Build the Pandas summary tables and return them in a dict of DataFrames.

    Uses ``groupby().agg``, ``pivot_table`` and ``value_counts`` as required.
    """
    labelled = df[df["bad_outcome"].notna()].copy()
    labelled["bad_outcome"] = labelled["bad_outcome"].astype(float)

    by_type = (labelled.groupby("animal_type")
               .agg(intakes=("animal_id", "size"),
                    bad_outcome_rate=("bad_outcome", "mean"),
                    median_stay_days=("length_of_stay_days", "median"),
                    median_age_years=("age_at_intake_years", "median"))
               .sort_values("intakes", ascending=False).round(3))

    by_condition = (labelled.groupby("intake_condition")
                    .agg(intakes=("animal_id", "size"), bad_outcome_rate=("bad_outcome", "mean"))
                    .sort_values("bad_outcome_rate", ascending=False).round(3))

    year_type = pd.pivot_table(df, index=df["intake_date"].dt.year, columns="animal_type",
                               values="animal_id", aggfunc="count", fill_value=0)
    year_type.index.name = "intake_year"

    outcome_counts = (df["outcome_type"].value_counts(dropna=False)
                      .rename_axis("outcome_type").reset_index(name="count"))
    outcome_counts["share_pct"] = (100 * outcome_counts["count"] / len(df)).round(2)

    # NumPy: percentiles of length of stay per animal type in one vectorised call.
    stay_pct = (df.groupby("animal_type")["length_of_stay_days"]
                .apply(lambda s: pd.Series(np.nanpercentile(s, [50, 90, 99]), index=["p50", "p90", "p99"]))
                .unstack().round(1))

    return {"by_animal_type": by_type, "by_intake_condition": by_condition,
            "intakes_by_year_and_type": year_type, "outcome_counts": outcome_counts,
            "stay_percentiles_by_type": stay_pct}


# ------------------------------------------------------------------ figures
def plot_monthly_intakes(df, path):
    """Line chart of intakes per month for cats, dogs and all other animals."""
    group = np.where(df["animal_type"].isin(["cat", "dog"]), df["animal_type"], "other")
    monthly = (df.assign(group=group)
               .groupby([pd.Grouper(key="intake_date", freq="MS"), "group"]).size()
               .unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"cat": BLUE, "dog": ORANGE, "other": "#1baf7a"}
    for name in ["cat", "dog", "other"]:
        ax.plot(monthly.index, monthly[name], color=colors[name], linewidth=2, label=name)
        ax.annotate(name, (monthly.index[-1], monthly[name].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color="#f0efec", zorder=0)
    ax.text(pd.Timestamp("2020-04-01"), ax.get_ylim()[1] * 0.92, "COVID-19 period", color=MUTED, fontsize=9)
    _style(ax, "Monthly intakes at Long Beach Animal Care Services, 2017-2024",
           "Intake month", "Intakes (animals per month)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bad_outcome_by_condition(tables, path):
    """Horizontal bar chart: share of animals that died / were euthanised, per intake condition."""
    data = tables["by_intake_condition"]
    data = data[data["intakes"] >= 50].sort_values("bad_outcome_rate")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data.index, data["bad_outcome_rate"] * 100, color=BLUE, height=0.7)
    for y, (rate, n) in enumerate(zip(data["bad_outcome_rate"], data["intakes"])):
        ax.text(rate * 100 + 1, y, f"{rate:.0%}  (n={n:,})", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, 110)
    _style(ax, "Share of animals that did not leave alive, by intake condition",
           "Died or euthanised (% of intakes)", "Intake condition")
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stay_by_type(df, path):
    """Box plot (log scale) of length of stay for the six most common animal types."""
    top = df["animal_type"].value_counts().index[:6]
    data = [df.loc[df["animal_type"] == t, "length_of_stay_days"].dropna() + 1 for t in top]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(data, tick_labels=list(top), showfliers=False, widths=0.5,
               medianprops={"color": ORANGE, "linewidth": 2},
               boxprops={"color": BLUE}, whiskerprops={"color": BLUE}, capprops={"color": BLUE})
    ax.set_yscale("log")
    _style(ax, "Length of stay by animal type (outliers hidden; birds and wildlife leave the same day)",
           "Animal type", "Days in shelter + 1 (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_age_group_outcome(df, path):
    """Grouped bars: bad-outcome rate by age group, cats vs dogs."""
    sub = df[df["animal_type"].isin(["cat", "dog"]) & df["bad_outcome"].notna()].copy()
    sub["bad_outcome"] = sub["bad_outcome"].astype(float)
    order = ["baby (<6m)", "young (6m-2y)", "adult (2-8y)", "senior (8y+)", "unknown"]
    table = sub.pivot_table(index="age_group", columns="animal_type", values="bad_outcome",
                            aggfunc="mean").reindex(order) * 100
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.19, table["cat"], width=0.36, color=BLUE, label="cat")
    ax.bar(x + 0.19, table["dog"], width=0.36, color=ORANGE, label="dog")
    ax.set_xticks(x, order)
    ax.legend(frameon=False)
    _style(ax, "Died or euthanised by age group: cats vs dogs",
           "Age at intake", "Died or euthanised (% of intakes)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_missing_before_after(before, after, path):
    """Bar chart comparing % missing per column before and after cleaning."""
    shared = before.index.intersection(after.index)  # only columns that exist in both
    cols = [c for c in shared if before[c] > 0 or after[c] > 0]
    b, a = before.reindex(cols).fillna(0), after.reindex(cols).fillna(0)
    order = b.sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y + 0.2, b[order], height=0.38, color="#b8b7b0", label="raw")
    ax.barh(y - 0.2, a[order], height=0.38, color=BLUE, label="cleaned")
    ax.set_yticks(y, order)
    ax.legend(frameon=False, loc="lower right")
    _style(ax, "Missing values per column, raw vs cleaned", "Missing (% of rows)", "Column")
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ SQLite
def run_sql(df, db_path=config.DB_FILE, sql_path=config.SQL_DIR / "outcome_by_intake_type.sql"):
    """Load the cleaned table plus a small lookup table into SQLite and run the query file.

    The lookup table groups the 12 intake types into 4 broader categories, so
    the query shows a real JOIN + GROUP BY.
    """
    lookup = pd.DataFrame({
        "intake_type": ["stray", "wildlife", "owner surrender", "welfare seized", "confiscate",
                        "return", "quarantine", "safe keep", "trap, neuter, return",
                        "euthanasia required", "foster", "adopted animal return"],
        "intake_group": ["found / stray", "wildlife", "owner gave up", "legal / enforcement",
                         "legal / enforcement", "owner gave up", "legal / enforcement",
                         "legal / enforcement", "found / stray", "owner gave up",
                         "owner gave up", "owner gave up"],
    })
    table = df.copy()
    for col in ("dob", "intake_date", "outcome_date"):
        table[col] = table[col].dt.strftime("%Y-%m-%d")
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as con:
        table.to_sql("intakes", con, index=False)
        lookup.to_sql("intake_type_lookup", con, index=False)
        query = sql_path.read_text(encoding="utf-8")
        result = pd.read_sql_query(query, con)
    return query, result


# ------------------------------------------------------------------ entry
def run_analysis(df, missing_before, missing_after):
    """Write all summary CSVs, figures and the SQL result. Returns the list of files."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    tables = summary_tables(df)
    for name, table in tables.items():
        path = config.REPORT_DIR / f"summary_{name}.csv"
        table.to_csv(path)
        written.append(path)

    figures = {
        "01_monthly_intakes.png": lambda p: plot_monthly_intakes(df, p),
        "02_bad_outcome_by_condition.png": lambda p: plot_bad_outcome_by_condition(tables, p),
        "03_length_of_stay_by_type.png": lambda p: plot_stay_by_type(df, p),
        "04_age_group_outcome_cat_dog.png": lambda p: plot_age_group_outcome(df, p),
        "05_missing_before_after.png": lambda p: plot_missing_before_after(missing_before, missing_after, p),
    }
    for name, draw in figures.items():
        path = config.FIG_DIR / name
        draw(path)
        written.append(path)

    _, result = run_sql(df)
    path = config.REPORT_DIR / "sql_outcome_by_intake_type.csv"
    result.to_csv(path, index=False)
    written.append(path)
    return written
