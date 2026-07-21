#!/usr/bin/env python3
"""
analyze_confidence_mc1.py — Confidence & calibration diagnostics, restricted to
the ``max_calls=1`` generate variations.

v0.6.0 of ``ollama-classifier`` changed the ``generate`` algorithm so that a
single constrained call already yields a full, internally consistent
probability distribution, and any additional calls (2..max_calls) only
*reproportion* probability mass within clusters of prefix-sharing labels
(mass-preserving; accuracy is monotone non-decreasing in ``max_calls``).
On the COICOP Division 01.2 task the seven subclass names share no token
prefixes, so no clusters ever fire: results are byte-identical across
``max_calls in {1, 3, 5, 8}`` and ``n_calls`` is always 1. The full 22-variation
analysis therefore carries 12 redundant rows.

This script reuses :mod:`analyze_confidence` and overrides its module-level
``VARIATIONS`` catalog *in place* so every downstream plotter / table builder
sees only the 10 distinct confidence-bearing variations (2 BART, 4 ``classify``,
4 ``generate`` at ``max_calls=1``). It also produces a new figure,
``generate_vs_classify``, that compares the two scoring strategies at matched
choice configurations.

Outputs overwrite, in experiment/:
  confidence_boxplots.{png,eps}
  scoregap_boxplots.{png,eps}
  selective_accuracy.{png,eps}
  coverage_curve.{png,eps}
  selective_accuracy_margin.{png,eps}
  coverage_curve_margin.{png,eps}
  calibration_curve.{png,eps}
  generate_vs_classify.{png,eps}    (NEW — replaces generate_sweep)
  ttest_results.{csv,txt}
  confidence_table.{csv,tex,txt}
  threshold_discrimination.{csv,tex,txt}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze_confidence as ac

# -----------------------------------------------------------------------------
# Override the shared catalog: keep every non-generate variation plus only the
# generate variations at max_calls == 1.
# -----------------------------------------------------------------------------
ac.VARIATIONS = [
    v for v in ac.VARIATIONS if v["method"] != "generate" or v["max_calls"] == 1
]

# 10 variations expected: 2 BART + 4 classify + 4 generate(mc=1).
assert len(ac.VARIATIONS) == 10, (
    f"Expected 10 mc=1-filtered variations, got {len(ac.VARIATIONS)}"
)


# =============================================================================
# New figure: generate (single call) vs classify (N calls), per choice config
# =============================================================================


def plot_generate_vs_classify(stats_df) -> None:
    """Grouped bar chart comparing generate(mc=1) and classify across configs.

    Four panels (Accuracy, Mean confidence, ECE, ROC-AUC of confidence), each
    with the four choice configurations on the x-axis and two bars per config
    (``classify`` exact vs ``generate`` single-call). The caption notes the call
    cost asymmetry: ``classify`` makes one call per label (7 without opt-out,
    8 with) per item, ``generate`` makes a single call per item.
    """
    panels = [
        ("Accuracy (%)", "Accuracy (%)", (0.0, 100.0)),
        ("Mean confidence", "Mean conf (all)", (0.0, 1.0)),
        ("Expected Calibration Error", "ECE", (0.0, 0.30)),
        ("ROC-AUC (confidence)", "AUC (conf)", (0.5, 1.0)),
    ]
    configs = ac.CONFIGS
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9), constrained_layout=True
    )
    axes = np.array(axes).ravel()

    classify_color = "#d62728"  # matches ac._FIXED_STYLE["Classify"]
    generate_color = ac._GEN_COLORS[1]  # max_calls=1 line color

    bar_width = 0.36
    x = np.arange(len(configs))

    for ax, (label, col, ylim) in zip(axes, panels):
        classify_vals = []
        generate_vals = []
        for config in configs:
            cl = stats_df[
                (stats_df["config"] == config)
                & (stats_df["method"] == "classify")
            ]
            gn = stats_df[
                (stats_df["config"] == config)
                & (stats_df["method"] == "generate")
            ]
            classify_vals.append(
                float(cl[col].iloc[0]) if len(cl) else float("nan")
            )
            generate_vals.append(
                float(gn[col].iloc[0]) if len(gn) else float("nan")
            )

        ax.bar(
            x - bar_width / 2,
            classify_vals,
            bar_width,
            color=classify_color,
            label="classify (exact, N calls)",
        )
        ax.bar(
            x + bar_width / 2,
            generate_vals,
            bar_width,
            color=generate_color,
            label="generate (1 call)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_title(label)
        # Annotate each bar with its value.
        for vals, offset in (
            (classify_vals, -bar_width / 2),
            (generate_vals, bar_width / 2),
        ):
            for xi, v in zip(x, vals):
                if np.isnan(v):
                    continue
                ax.text(
                    xi + offset,
                    v + (ylim[1] - ylim[0]) * 0.015,
                    f"{v:.2f}" if ylim[1] <= 1.0 else f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=11,
        frameon=True,
        title="Scoring strategy",
        title_fontsize=11,
    )
    fig.suptitle(
        "classify vs generate (max_calls=1) on COICOP Division 01.2",
        fontsize=15,
    )
    ac._save_fig(fig, "generate_vs_classify")


# =============================================================================
# Main (reuses the shared figure/table pipeline; only the generate figure differs)
# =============================================================================


def main() -> None:
    df = ac.pd.read_excel(ac.RESULTS_XLSX, sheet_name=ac.SHEET)
    long_df = ac.build_long_frame(df)

    stats_df, conf_pvals, gap_pvals = ac.compute_stats(long_df)

    # Shared standard figures (boxplots, threshold curves, calibration) and
    # tables are emitted by the base module, so this driver cannot drift from it.
    ac.plot_standard_figures(long_df, stats_df, conf_pvals, gap_pvals)

    # NEW: generate-vs-classify comparison (replaces plot_generate_sweep).
    plot_generate_vs_classify(stats_df)

    ac.save_all_tables(long_df, stats_df)

    print(
        f"Generated {len(ac.VARIATIONS)} variations "
        f"(BART + classify + generate mc=1 only)."
    )
    print(f"Outputs written to: {ac.EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
