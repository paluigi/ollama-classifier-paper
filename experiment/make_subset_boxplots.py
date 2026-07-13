#!/usr/bin/env python3
"""Generate presentation-friendly boxplots: BART + classify only (no generate)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyze_confidence import (  # noqa: E402
    VARIATIONS,
    _p_label,
    _save_fig,
    build_long_frame,
    compute_stats,
)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

RESULTS_XLSX = Path(__file__).parent / "results.xlsx"
SHEET = "Detailed Results"

SUBSET = [v for v in VARIATIONS if v["method"] != "generate"]


def plot_boxplots_subset(
    long_df: pd.DataFrame,
    value_col: str,
    pvals: dict,
    ylabel: str,
    title: str,
    stem: str,
) -> None:
    plot_df = long_df.dropna(subset=[value_col]).copy()
    plot_df["outcome"] = plot_df["correct"].map(
        lambda c: "Correct" if c else "Incorrect"
    )

    present = [v for v in SUBSET if v["display"] in set(plot_df["methodology"])]
    n = len(present)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.8 * n_rows),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.array(axes).ravel()

    for ax, var in zip(axes, present):
        sub = plot_df[plot_df["methodology"] == var["display"]]
        sns.boxplot(
            data=sub,
            x="outcome",
            y=value_col,
            order=["Correct", "Incorrect"],
            hue="outcome",
            palette={"Correct": "#4C9F70", "Incorrect": "#D1495B"},
            width=0.55,
            fliersize=2,
            ax=ax,
            legend=False,
        )
        sns.stripplot(
            data=sub,
            x="outcome",
            y=value_col,
            order=["Correct", "Incorrect"],
            color="#888888",
            size=2,
            jitter=True,
            ax=ax,
        )
        ax.set_title(
            f"{var['display']}\nN = {len(sub)}   ({_p_label(pvals.get(var['display'], float('nan')))})",
            fontsize=11,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.set_ylim(-0.02, 1.02)

    for ax in axes[len(present):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14)
    _save_fig(fig, stem)


def main() -> None:
    df = pd.read_excel(RESULTS_XLSX, sheet_name=SHEET)
    long_df = build_long_frame(df)
    _, conf_pvals, gap_pvals = compute_stats(long_df)

    plot_boxplots_subset(
        long_df,
        value_col="confidence",
        pvals=conf_pvals,
        ylabel="Confidence",
        title="Confidence: Correct vs Incorrect (BART & classify)",
        stem="confidence_boxplots_subset",
    )
    plot_boxplots_subset(
        long_df,
        value_col="score_gap",
        pvals=gap_pvals,
        ylabel="Score gap (top - 2nd prob.)",
        title="Score gap: Correct vs Incorrect (BART & classify)",
        stem="scoregap_boxplots_subset",
    )
    print("Subset boxplots written.")


if __name__ == "__main__":
    main()
