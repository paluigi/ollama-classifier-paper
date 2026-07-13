#!/usr/bin/env python3
"""
analyze_confidence.py — Confidence-diagnostics for the zero-shot benchmark.

Reads the "Detailed Results" sheet of ``experiment/results.xlsx`` (produced by
``experiment.py``) and produces four figures plus a t-test table:

  1. Boxplots of confidence split by Correct/Incorrect, faceted by methodology.
  2. One-tailed Welch t-tests per methodology (H1: incorrect < correct).
  3. Selective accuracy: accuracy of the subset whose confidence >= threshold,
     as a function of the threshold (one line per methodology).
  4. Coverage: percentage of products whose confidence >= threshold, as a
     function of the threshold (one line per methodology).

scikit-llm is excluded (no confidence scores). Opt-out ("None of the above")
and ERROR predictions are dropped before every analysis, mirroring
``compute_confidence_analysis`` in ``experiment.py``.

Outputs (all in experiment/):
  confidence_boxplots.{png,eps}
  selective_accuracy.{png,eps}
  coverage_curve.{png,eps}
  ttest_results.csv
  ttest_results.txt
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from variations import VARIATION_SPECS

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_XLSX = EXPERIMENT_DIR / "results.xlsx"
SHEET = "Detailed Results"

OPT_OUT_LABEL = "None of the above"
ERROR_LABEL = "ERROR"
DROP_CODES = {OPT_OUT_LABEL, ERROR_LABEL}

DPI = 300

# Confidence-bearing methodologies only. scikit-llm is excluded (no confidence
# scores). The keys and run order are imported from ``variations.py`` — the same
# shared catalog ``experiment.py`` writes — so the two scripts cannot drift
# apart. ``build_long_frame`` still tolerates a partial ``results.xlsx``
# (missing columns are skipped) so the analysis can run before the full sweep
# completes.


def _spec_display(spec) -> str:
    """Legend label for a variation (compact generate suffix for plots)."""
    name = spec.base_display()
    if spec.method == "generate":
        name = f"{name} [gen mc={spec.max_calls}]"
    return name


VARIATIONS: list[dict] = [
    {
        "display": _spec_display(spec),
        "key": spec.key,
        "has_opt_out": spec.has_opt_out,
    }
    for spec in VARIATION_SPECS
    if spec.classifier != "skllm"
]

THRESHOLDS = np.linspace(0.0, 1.0, 101)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("husl", n_colors=len(VARIATIONS))


# =============================================================================
# Load & reshape
# =============================================================================


def build_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format frame: methodology | confidence | correct (bool).

    Excludes opt-out/ERROR predictions and rows with missing confidence.
    Methodologies whose confidence column is absent in ``df`` are skipped, so
    the analysis can run against a partial ``results.xlsx``.
    """
    rows = []
    for var in VARIATIONS:
        conf_col = f"{var['key']}_conf"
        code_col = f"{var['key']}_code"
        if conf_col not in df.columns or code_col not in df.columns:
            continue
        code = df[code_col]
        conf = df[conf_col]
        gt = df["ground_truth"]

        valid = conf.notna() & ~code.isin(DROP_CODES)
        correct = (code[valid] == gt[valid]).to_numpy()

        rows.append(
            pd.DataFrame(
                {
                    "methodology": var["display"],
                    "confidence": conf[valid].to_numpy(dtype=float),
                    "correct": correct,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def correctness_label(is_correct: bool) -> str:
    return "Correct" if is_correct else "Incorrect"


# =============================================================================
# Task 1 — Boxplots: confidence by Correct/Incorrect, faceted by methodology
# =============================================================================


def plot_boxplots(long_df: pd.DataFrame, pvals: dict[str, float]) -> None:
    plot_df = long_df.assign(outcome=long_df["correct"].map(correctness_label))

    present = [v for v in VARIATIONS if v["display"] in set(plot_df["methodology"])]
    n = len(present)
    n_cols = 3
    n_rows = math.ceil(n / n_cols) if n else 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 3.6 * n_rows),
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()

    for ax, var in zip(axes, present):
        name = var["display"]
        sub = plot_df[plot_df["methodology"] == name]
        sns.boxplot(
            data=sub,
            x="outcome",
            y="confidence",
            order=["Correct", "Incorrect"],
            hue="outcome",
            palette={"Correct": "#4C9F70", "Incorrect": "#D1495B"},
            width=0.55,
            fliersize=2,
            ax=ax,
            legend=False,
        )
        # Solid (non-transparent) points keep the EPS fully vector and small;
        # overlaps darken naturally to convey density.
        sns.stripplot(
            data=sub,
            x="outcome",
            y="confidence",
            order=["Correct", "Incorrect"],
            color="#888888",
            size=2,
            jitter=True,
            ax=ax,
        )

        p = pvals[name]
        psig = p if p >= 0.001 else "< 0.001"
        n = len(sub)
        ax.set_title(f"{name}\nN = {n}   (p {psig})", fontsize=15)
        ax.set_xlabel("")
        ax.set_ylabel("Confidence" if ax is axes[0] else "")
        ax.set_ylim(-0.02, 1.02)

    for ax in axes[len(present) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Confidence distribution: Correct vs Incorrect predictions",
        fontsize=18,
        y=1.02,
    )

    for ext in ("png", "eps"):
        fig.savefig(
            EXPERIMENT_DIR / f"confidence_boxplots.{ext}",
            dpi=DPI,
            bbox_inches="tight",
        )
    plt.close(fig)


# =============================================================================
# Task 2 — One-tailed Welch t-tests (H1: incorrect mean < correct mean)
# =============================================================================


def run_ttests(long_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Welch t-test, alternative = correct > incorrect (i.e. incorrect lower)."""
    rows = []
    pvals: dict[str, float] = {}
    for var in VARIATIONS:
        name = var["display"]
        sub = long_df[long_df["methodology"] == name]
        correct = sub.loc[sub["correct"], "confidence"].to_numpy(dtype=float)
        incorrect = sub.loc[~sub["correct"], "confidence"].to_numpy(dtype=float)

        if len(correct) < 2 or len(incorrect) < 2:
            t_stat = p_value = np.nan
        else:
            t_stat, p_value = stats.ttest_ind(
                correct, incorrect, alternative="greater", equal_var=False
            )

        pvals[name] = p_value
        rows.append(
            {
                "Methodology": name,
                "N correct": int(len(correct)),
                "N incorrect": int(len(incorrect)),
                "Mean conf (correct)": round(float(np.mean(correct)), 4),
                "Mean conf (incorrect)": round(float(np.mean(incorrect)), 4),
                "Mean diff (C-I)": round(
                    float(np.mean(correct) - np.mean(incorrect)), 4
                ),
                "t-statistic": (
                    round(float(t_stat), 4) if not np.isnan(t_stat) else None
                ),
                "p-value (one-tailed)": (
                    round(float(p_value), 6) if not np.isnan(p_value) else None
                ),
                "Significant (alpha=0.05)": (
                    bool(p_value < 0.05) if not np.isnan(p_value) else False
                ),
            }
        )

    return pd.DataFrame(rows), pvals


def save_ttest_results(ttest_df: pd.DataFrame) -> None:
    ttest_df.to_csv(EXPERIMENT_DIR / "ttest_results.csv", index=False)

    with open(EXPERIMENT_DIR / "ttest_results.txt", "w") as f:
        f.write("One-tailed Welch t-tests: confidence Correct vs Incorrect\n")
        f.write("H1: mean(confidence | Incorrect) < mean(confidence | Correct)\n")
        f.write("=" * 72 + "\n\n")
        f.write(ttest_df.to_string(index=False))
        f.write("\n")


# =============================================================================
# Task 3 — Selective accuracy vs confidence threshold
# =============================================================================


def compute_threshold_curves(
    long_df: pd.DataFrame,
) -> tuple[dict[str, list], dict[str, list]]:
    """Return accuracy curves and coverage curves per methodology.

    accuracy(t) = fraction correct among items with confidence >= t.
    coverage(t) = % of valid items with confidence >= t.

    Each curve stops at the first threshold that leaves no items, so the line
    never plots accuracy over an empty subset.
    """
    acc_curves: dict[str, list] = {}
    cov_curves: dict[str, list] = {}
    for var in VARIATIONS:
        name = var["display"]
        conf = long_df.loc[long_df["methodology"] == name, "confidence"].to_numpy(
            dtype=float
        )
        correct = long_df.loc[long_df["methodology"] == name, "correct"].to_numpy()
        n_valid = len(conf)

        if n_valid == 0:
            acc_curves[name] = ([], [])
            cov_curves[name] = ([], [])
            continue

        xs_acc, ys_acc = [], []
        xs_cov, ys_cov = [], []
        for t in THRESHOLDS:
            mask = conf >= t
            ys_cov.append(float(mask.sum()) / n_valid * 100)
            xs_cov.append(float(t))
            if mask.sum() == 0:
                break
            xs_acc.append(float(t))
            ys_acc.append(float(correct[mask].mean()))

        acc_curves[name] = (xs_acc, ys_acc)
        cov_curves[name] = (xs_cov, ys_cov)
    return acc_curves, cov_curves


def plot_line_curves(
    curves: dict[str, tuple[list, list]],
    ylabel: str,
    title: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for var, color in zip(VARIATIONS, PALETTE):
        name = var["display"]
        xs, ys = curves[name]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=3,
            linewidth=2,
            label=name,
            color=color,
        )

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.01, 1.01)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0.0,
        fontsize=11,
        title="Methodology",
    )

    for ext in ("png", "eps"):
        fig.savefig(
            EXPERIMENT_DIR / f"{filename}.{ext}",
            dpi=DPI,
            bbox_inches="tight",
        )
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    df = pd.read_excel(RESULTS_XLSX, sheet_name=SHEET)
    long_df = build_long_frame(df)

    ttest_df, pvals = run_ttests(long_df)
    save_ttest_results(ttest_df)

    acc_curves, cov_curves = compute_threshold_curves(long_df)

    plot_boxplots(long_df, pvals)
    plot_line_curves(
        acc_curves,
        ylabel="Accuracy (correct / items above threshold)",
        title="Selective accuracy vs confidence threshold",
        filename="selective_accuracy",
        ylim=(0.0, 1.02),
    )
    plot_line_curves(
        cov_curves,
        ylabel="Coverage (% of products above threshold)",
        title="Coverage vs confidence threshold",
        filename="coverage_curve",
        ylim=(0.0, 100.5),
    )

    print("t-test results:")
    print(ttest_df.to_string(index=False))
    print(f"\nOutputs written to: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
