#!/usr/bin/env python3
"""
analyze_confidence.py — Confidence & calibration diagnostics for the
zero-shot classification benchmark.

Reads the "Detailed Results" sheet of ``experiment/results.xlsx`` (produced by
``experiment.py``) and produces a comprehensive post-hoc analysis over all 22
confidence-bearing variations (BART ×2, ollama-classifier ``classify`` ×4, and
``generate`` ×16 = 4 choice-configs × 4 ``max_calls`` budgets). scikit-llm is
excluded (label-only; no confidence scores).

The analysis treats **confidence** and the **score gap** (top − second class
probability) as parallel per-item metrics. For each, it compares correct vs
incorrect predictions with a one-tailed Welch *t*-test, reports Cohen's *d*,
and applies Bonferroni and Benjamini–Hochberg multiple-comparison correction
within each metric family (22 tests each).

Outputs (all in experiment/):
  confidence_boxplots.{png,eps}   — confidence by Correct/Incorrect (22 facets)
  scoregap_boxplots.{png,eps}     — score gap by Correct/Incorrect (22 facets)
  selective_accuracy.{png,eps}    — accuracy vs confidence threshold (2×2 by config)
  coverage_curve.{png,eps}        — coverage vs confidence threshold (2×2 by config)
  selective_accuracy_margin.{png,eps} — accuracy vs score-gap (margin) threshold (2×2)
  coverage_curve_margin.{png,eps}    — coverage vs score-gap (margin) threshold (2×2)
  calibration_curve.{png,eps}     — reliability diagram + per-variation ECE (2×2)
  generate_sweep.{png,eps}        — accuracy/conf/ECE/AUC vs max_calls (generate only)
  ttest_results.{csv,txt}         — Welch t-tests for both metrics + correction
  confidence_table.{csv,tex,txt}  — combined, paper-ready per-variation table
  threshold_discrimination.{csv,tex,txt} — Youden-optimal confidence/margin thresholds
                                       (AUC, threshold, selective accuracy, coverage)

Opt-out ("None of the above") and ERROR predictions are dropped before every
analysis, mirroring ``compute_confidence_analysis`` in ``experiment.py``.
``build_long_frame`` tolerates a partial ``results.xlsx`` (missing columns are
skipped) so the analysis can run before the full sweep completes.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

from variations import MAX_CALLS, VARIATION_SPECS

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
N_BINS = 10  # equal-width bins for ECE / reliability
ALPHA = 0.05

sns.set_theme(style="whitegrid", context="talk")

# Confidence-bearing methodologies only. scikit-llm is excluded (no confidence
# scores). Keys and run order come from ``variations.py`` — the same shared
# catalog ``experiment.py`` writes — so the two scripts cannot drift apart.


def _config_label(spec) -> str:
    """Unified 4-level facet: BART and Ollama share the same config slots."""
    if spec.flavor == "names_with_desc":
        return "Desc. + opt-out" if spec.has_opt_out else "Names + desc."
    return "Names + opt-out" if spec.has_opt_out else "Names only"


CONFIGS = ["Names only", "Names + opt-out", "Names + desc.", "Desc. + opt-out"]


def _display(spec) -> str:
    """Compact legend / facet label for a variation."""
    name = spec.base_display()
    if spec.method == "generate":
        name = f"{name} [gen mc={spec.max_calls}]"
    return name


VARIATIONS: list[dict] = [
    {
        "display": _display(spec),
        "key": spec.key,
        "config": _config_label(spec),
        "classifier": spec.classifier,
        "method": spec.method,
        "max_calls": spec.max_calls,
        "has_opt_out": spec.has_opt_out,
    }
    for spec in VARIATION_SPECS
    if spec.classifier != "skllm"
]


# --- Consistent line encoding across all figures ---------------------------
# BART and classify get fixed distinct colours; the four generate call-budgets
# share a sequential colormap so the budget trend is legible within each panel.

_GEN_COLORS = {
    mc: c
    for mc, c in zip(MAX_CALLS, plt.cm.viridis(np.linspace(0.15, 0.9, len(MAX_CALLS))))
}
_FIXED_STYLE = {
    "BART": {"color": "#1f77b4", "linestyle": "--", "marker": "s"},
    "Classify": {"color": "#d62728", "linestyle": "-", "marker": "^"},
}


def _method_type(var: dict) -> tuple[str, int | None]:
    if var["classifier"] == "bart":
        return "BART", None
    if var["method"] == "classify":
        return "Classify", None
    return "Generate", var["max_calls"]


def _line_props(var: dict) -> dict:
    mtype, mc = _method_type(var)
    if mtype == "Generate":
        return {
            "color": _GEN_COLORS[mc],
            "linestyle": "-",
            "marker": "o",
            "label": f"Generate (mc={mc})",
        }
    style = _FIXED_STYLE[mtype]
    return {**style, "label": mtype}


def _legend_handles() -> list[Line2D]:
    """One proxy handle per method type present, in run order."""
    seen: set[str] = set()
    handles: list[Line2D] = []
    for var in VARIATIONS:
        props = _line_props(var)
        if props["label"] in seen:
            continue
        seen.add(props["label"])
        handles.append(
            Line2D(
                [0],
                [0],
                color=props["color"],
                marker=props["marker"],
                linestyle=props["linestyle"],
                markersize=7,
                linewidth=2,
                label=props["label"],
            )
        )
    return handles


def _add_right_legend(fig) -> None:
    fig.legend(
        handles=_legend_handles(),
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0.0,
        fontsize=11,
        title="Method",
        frameon=False,
    )


# =============================================================================
# Statistic helpers (numpy / scipy only — no scikit-learn dependency)
# =============================================================================


def _score_gap_from_probs(probs: dict) -> float | None:
    """Margin of victory: highest minus second-highest class probability.

    Mirrors ``experiment.py:_score_gap_from_probs`` (rounded to 4 dp). Computed
    for every valid item; the all-items mean then reproduces the experiment's
    Confidence Summary "Mean Score Gap" (experiment.py averages over its full
    valid index, not incorrect-only despite its comment).
    """
    if not isinstance(probs, dict) or len(probs) < 2:
        return None
    vals = sorted((float(v) for v in probs.values()), reverse=True)
    return round(vals[0] - vals[1], 4)


def _parse_probs(val) -> dict:
    if pd.isna(val) or val in ("", "{}"):
        return {}
    try:
        parsed = json.loads(val)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _welch_greater(a, b) -> tuple[float, float]:
    """One-tailed Welch t-test, H1: mean(a) > mean(b). NaN if too few values."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, alternative="greater", equal_var=False)
    return float(t), float(p)


def _cohens_d(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def _auc_ranksum(score, label) -> float:
    """ROC-AUC via the Mann–Whitney rank-sum (midrank ties)."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label).astype(bool)
    mask = ~np.isnan(score)
    score, label = score[mask], label[mask]
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _youden_threshold(score, correct) -> tuple[float, float, float, float]:
    """Youden-optimal threshold on ``score`` for separating correct from incorrect.

    Sweeps candidate thresholds (unique scores); maximises
    ``J = TPR - FPR``. Returns ``(threshold, selective_accuracy, coverage_pct, J)``
    where selective accuracy is the accuracy among items with score >= threshold
    and coverage is the percentage of items retained. NaN if the score cannot
    discriminate (one class empty or all values missing).
    """
    score = np.asarray(score, dtype=float)
    correct = np.asarray(correct).astype(bool)
    mask = ~np.isnan(score)
    score, correct = score[mask], correct[mask]
    n_pos = int(correct.sum())
    n_neg = int((~correct).sum())
    n = len(score)
    if n == 0 or n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    best = None  # (J, threshold, selective_accuracy, coverage_pct)
    for t in np.unique(score):
        pred = score >= t
        retained = int(pred.sum())
        if retained == 0:
            continue
        tp = int((pred & correct).sum())
        tpr = tp / n_pos
        fpr = int((pred & ~correct).sum()) / n_neg
        j = tpr - fpr
        if best is None or j > best[0]:
            best = (j, float(t), tp / retained, retained / n * 100)
    if best is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return best[1], best[2], best[3], best[0]


def _brier(conf, correct) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    mask = ~np.isnan(conf)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((conf[mask] - correct[mask]) ** 2))


def _ece(conf, correct, n_bins: int = N_BINS) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    mask = ~np.isnan(conf)
    conf, correct = conf[mask], correct[mask]
    n = len(conf)
    if n == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            in_bin = (conf >= edges[i]) & (conf < edges[i + 1])
        cnt = int(in_bin.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def _reliability_points(conf, correct, n_bins: int = N_BINS) -> tuple[list, list]:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    mask = ~np.isnan(conf)
    conf, correct = conf[mask], correct[mask]
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            in_bin = (conf >= edges[i]) & (conf < edges[i + 1])
        if in_bin.sum() == 0:
            continue
        xs.append(float(conf[in_bin].mean()))
        ys.append(float(correct[in_bin].mean()))
    return xs, ys


def _pointbiserial(correct, conf) -> tuple[float, float]:
    correct = np.asarray(correct, dtype=float)
    conf = np.asarray(conf, dtype=float)
    mask = ~np.isnan(conf)
    c, x = correct[mask], conf[mask]
    if len(c) < 3 or np.unique(c).size < 2:
        return float("nan"), float("nan")
    r, p = stats.pointbiserialr(c, x)
    return float(r), float(p)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # monotone from largest down
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


# =============================================================================
# Load & reshape
# =============================================================================


def _read_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df), index=df.index)


def build_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rich long-format frame: one row per (item, variation).

    Carries every field the downstream analyses need: confidence, correctness,
    the per-item score gap (computed for ALL valid items so it can be split by
    correct/incorrect), plus the generate-specific ``approx`` / ``ncalls``
    flags. Excludes opt-out/ERROR predictions and rows with missing confidence;
    variations whose columns are absent in ``df`` are skipped (partial-file
    tolerance).
    """
    gt = df["ground_truth"]
    frames = []
    for var in VARIATIONS:
        k = var["key"]
        conf_col, code_col = f"{k}_conf", f"{k}_code"
        if conf_col not in df.columns or code_col not in df.columns:
            continue
        code = df[code_col]
        conf = df[conf_col]
        probs = _read_col(df, f"{k}_probs")
        approx = _read_col(df, f"{k}_approx")
        ncalls = _read_col(df, f"{k}_ncalls")

        valid = conf.notna() & ~code.isin(DROP_CODES)
        idx = valid[valid].index

        score_gaps = []
        for i in idx:
            score_gaps.append(_score_gap_from_probs(_parse_probs(probs.loc[i])))

        frames.append(
            pd.DataFrame(
                {
                    "methodology": var["display"],
                    "key": k,
                    "config": var["config"],
                    "classifier": var["classifier"],
                    "method": var["method"],
                    "max_calls": var["max_calls"],
                    "confidence": conf.loc[idx].to_numpy(dtype=float),
                    "score_gap": score_gaps,
                    "correct": (code.loc[idx] == gt.loc[idx]).to_numpy(),
                    "approx": approx.loc[idx].to_numpy(),
                    "ncalls": ncalls.loc[idx].to_numpy(),
                },
                index=range(len(idx)),
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "methodology",
                "key",
                "config",
                "classifier",
                "method",
                "max_calls",
                "confidence",
                "score_gap",
                "correct",
                "approx",
                "ncalls",
            ]
        )
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Per-variation statistics
# =============================================================================


def _variation_subset(long_df: pd.DataFrame, name: str) -> pd.DataFrame:
    return long_df[long_df["methodology"] == name]


def compute_stats(long_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Per-variation raw statistics for both metrics + ranking + calibration.

    Returns (stats_df, conf_pvals, gap_pvals) where the two pval dicts map
    methodology -> raw one-tailed p (for the multiple-comparison pass and for
    boxplot annotations).
    """
    rows = []
    conf_pvals: dict[str, float] = {}
    gap_pvals: dict[str, float] = {}
    for var in VARIATIONS:
        name = var["display"]
        sub = _variation_subset(long_df, name)
        if len(sub) == 0:
            continue
        conf = sub["confidence"].to_numpy(dtype=float)
        correct = sub["correct"].to_numpy()
        gap = sub["score_gap"].to_numpy(dtype=float)  # may contain NaN

        conf_c = conf[correct]
        conf_i = conf[~correct]
        # score gap: only where defined; report overall + correct/incorrect split.
        # NB: experiment.py's Confidence-Summary "Mean Score Gap" is also the
        # all-items mean (its loop iterates the full valid index), so "Mean gap
        # (all)" is the value verifiable against summary.txt.
        gap_valid = ~np.isnan(gap)
        gap_all = gap[gap_valid]
        gap_c = gap[correct & gap_valid]
        gap_i = gap[(~correct) & gap_valid]

        t_conf, p_conf = _welch_greater(conf_c, conf_i)
        t_gap, p_gap = _welch_greater(gap_c, gap_i)
        conf_pvals[name] = p_conf
        gap_pvals[name] = p_gap

        corr_r, _ = _pointbiserial(correct, conf)
        auc_conf = _auc_ranksum(conf, correct)
        auc_gap = _auc_ranksum(gap, correct)
        brier = _brier(conf, correct)
        ece = _ece(conf, correct)

        rows.append(
            {
                "Variation": name,
                "config": var["config"],
                "classifier": var["classifier"],
                "method": var["method"],
                "max_calls": var["max_calls"],
                "N": int(len(sub)),
                "N correct": int(correct.sum()),
                "N incorrect": int((~correct).sum()),
                "Accuracy (%)": (
                    float(correct.mean() * 100) if len(correct) else float("nan")
                ),
                "Mean conf (all)": float(np.mean(conf)) if len(conf) else float("nan"),
                "Mean conf (correct)": (
                    float(np.mean(conf_c)) if len(conf_c) else float("nan")
                ),
                "Mean conf (incorrect)": (
                    float(np.mean(conf_i)) if len(conf_i) else float("nan")
                ),
                "Mean diff (conf)": (
                    float(np.mean(conf_c) - np.mean(conf_i))
                    if (len(conf_c) and len(conf_i))
                    else float("nan")
                ),
                "Welch t (conf)": t_conf,
                "p (conf)": p_conf,
                "Cohen's d (conf)": _cohens_d(conf_c, conf_i),
                "N gap correct": int(len(gap_c)),
                "N gap incorrect": int(len(gap_i)),
                "Mean gap (all)": (
                    float(np.mean(gap_all)) if len(gap_all) else float("nan")
                ),
                "Mean gap (correct)": (
                    float(np.mean(gap_c)) if len(gap_c) else float("nan")
                ),
                "Mean gap (incorrect)": (
                    float(np.mean(gap_i)) if len(gap_i) else float("nan")
                ),
                "Mean diff (gap)": (
                    float(np.mean(gap_c) - np.mean(gap_i))
                    if (len(gap_c) and len(gap_i))
                    else float("nan")
                ),
                "Welch t (gap)": t_gap,
                "p (gap)": p_gap,
                "Cohen's d (gap)": _cohens_d(gap_c, gap_i),
                "Corr r": corr_r,
                "AUC (conf)": auc_conf,
                "AUC (gap)": auc_gap,
                "Brier": brier,
                "ECE": ece,
            }
        )

    stats_df = pd.DataFrame(rows)

    # Multiple-comparison correction within each family (Bonferroni + BH-FDR).
    def _correct(col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
        p = stats_df[col].to_numpy(dtype=float)
        k = int(np.sum(~np.isnan(p)))
        idx = stats_df.index
        bonf = pd.Series(p < (ALPHA / k if k else ALPHA), index=idx)
        adj = _bh_fdr(p)
        bh = pd.Series(adj < ALPHA, index=idx)
        return bonf, bh, pd.Series(adj, index=idx)

    bonf_conf, sig_bh_conf, adj_p_conf = _correct("p (conf)")
    bonf_gap, sig_bh_gap, adj_p_gap = _correct("p (gap)")
    stats_df["Sig conf (uncorrected)"] = stats_df["p (conf)"] < ALPHA
    stats_df["Sig conf (Bonferroni)"] = bonf_conf
    stats_df["Sig conf (BH-FDR)"] = sig_bh_conf.values
    stats_df["p adj (conf, BH)"] = adj_p_conf.values
    stats_df["Sig gap (uncorrected)"] = stats_df["p (gap)"] < ALPHA
    stats_df["Sig gap (Bonferroni)"] = bonf_gap
    stats_df["Sig gap (BH-FDR)"] = sig_bh_gap.values
    stats_df["p adj (gap, BH)"] = adj_p_gap.values

    return stats_df, conf_pvals, gap_pvals


# =============================================================================
# Plots
# =============================================================================


def _p_label(p: float) -> str:
    if np.isnan(p):
        return "p n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _save_fig(fig, stem: str) -> None:
    for ext in ("png", "eps"):
        fig.savefig(EXPERIMENT_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(
    long_df: pd.DataFrame,
    value_col: str,
    pvals: dict[str, float],
    ylabel: str,
    title: str,
    stem: str,
) -> None:
    """22-facet boxplot of ``value_col`` split by Correct/Incorrect."""
    plot_df = long_df.dropna(subset=[value_col]).copy()
    plot_df["outcome"] = plot_df["correct"].map(
        lambda c: "Correct" if c else "Incorrect"
    )

    present = [v for v in VARIATIONS if v["display"] in set(plot_df["methodology"])]
    n = len(present)
    n_cols = 4
    n_rows = math.ceil(n / n_cols) if n else 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 3.2 * n_rows),
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
            fontsize=12,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.set_ylim(-0.02, 1.02)

    for ax in axes[len(present) :]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=18, y=1.01)
    _save_fig(fig, stem)


def _threshold_curves(
    long_df: pd.DataFrame, name: str, value_col: str = "confidence"
) -> dict:
    """Accuracy + coverage curves vs a threshold on ``value_col`` for one variation.

    ``value_col`` is ``"confidence"`` (predicted-label probability) or ``"score_gap"``
    (top - second probability, the margin). Rows with missing values are dropped.
    """
    sub = _variation_subset(long_df, name)
    val = sub[value_col].to_numpy(dtype=float)
    correct = sub["correct"].to_numpy(dtype=float)
    mask = ~np.isnan(val)
    val, correct = val[mask], correct[mask]
    n = len(val)
    thresholds = np.linspace(0.0, 1.0, 101)
    xs_acc, ys_acc, xs_cov, ys_cov = [], [], [], []
    for t in thresholds:
        m = val >= t
        ys_cov.append(float(m.sum()) / n * 100)
        xs_cov.append(float(t))
        if m.sum() == 0:
            break
        xs_acc.append(float(t))
        ys_acc.append(float(correct[m].mean()))
    return {"acc": (xs_acc, ys_acc), "cov": (xs_cov, ys_cov)}


def plot_faceted_threshold(
    long_df: pd.DataFrame,
    which: str,
    value_col: str,
    ylabel: str,
    title: str,
    stem: str,
    ylim: tuple[float, float],
    xlabel: str,
) -> None:
    """2×2 faceted threshold curve (selective accuracy or coverage) by config.

    Thresholds on ``value_col`` (``"confidence"`` or ``"score_gap"``).
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(15, 11), sharex=True, sharey=True, constrained_layout=True
    )
    axes = np.array(axes).ravel()
    for ax, config in zip(axes, CONFIGS):
        for var in VARIATIONS:
            if var["config"] != config:
                continue
            name = var["display"]
            if _variation_subset(long_df, name).empty:
                continue
            curves = _threshold_curves(long_df, name, value_col)
            xs, ys = curves[which]
            props = _line_props(var)
            ax.plot(
                xs,
                ys,
                marker=props["marker"],
                markersize=3,
                linewidth=2,
                color=props["color"],
                linestyle=props["linestyle"],
            )
        ax.set_title(config)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(*ylim)

    # Only label the outer axes to avoid repetition.
    for ax in axes[2:]:
        ax.set_xlabel(xlabel)
    for ax in axes[::2]:
        ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=15)
    _add_right_legend(fig)
    _save_fig(fig, stem)


def plot_calibration(long_df: pd.DataFrame) -> None:
    """Reliability diagram: empirical accuracy vs mean confidence, 2×2 by config."""
    fig, axes = plt.subplots(
        2, 2, figsize=(15, 11), sharex=True, sharey=True, constrained_layout=True
    )
    axes = np.array(axes).ravel()
    for ax, config in zip(axes, CONFIGS):
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect")
        for var in VARIATIONS:
            if var["config"] != config:
                continue
            sub = _variation_subset(long_df, var["display"])
            if sub.empty:
                continue
            xs, ys = _reliability_points(sub["confidence"], sub["correct"])
            props = _line_props(var)
            ax.plot(
                xs,
                ys,
                marker=props["marker"],
                markersize=6,
                linewidth=2,
                color=props["color"],
                linestyle=props["linestyle"],
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Mean confidence in bin")
        ax.set_ylabel("Empirical accuracy in bin")
        ax.set_title(config)

    fig.suptitle("Reliability (calibration) curves", fontsize=18, y=1.01)
    _add_right_legend(fig)
    _save_fig(fig, "calibration_curve")


def plot_generate_sweep(stats_df: pd.DataFrame) -> None:
    """accuracy / mean confidence / ECE / AUC vs max_calls, one line per config."""
    gen = stats_df[stats_df["method"] == "generate"].copy()
    if gen.empty:
        return

    metrics = [
        ("Accuracy (%)", "Accuracy (%)"),
        ("Mean confidence", "Mean conf (all)"),
        ("ECE", "ECE"),
        ("AUC", "AUC (conf)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = np.array(axes).ravel()

    for ax, (label, col) in zip(axes, metrics):
        for config in CONFIGS:
            sub = gen[gen["config"] == config].sort_values("max_calls")
            if sub.empty:
                continue
            xs = sub["max_calls"].to_numpy()
            ys = sub[col].to_numpy(dtype=float)
            ax.plot(xs, ys, marker="o", linewidth=2, label=config)
        # Boundary between the approximate regime (max_calls <= 5, all items
        # approximate) and the exact point (max_calls = 8).
        ax.axvline(6.5, color="0.5", linestyle=":", linewidth=1)
        ax.set_xticks(MAX_CALLS)
        ax.set_title(label)

    axes[0].text(
        3.0,
        axes[0].get_ylim()[1],
        "approximate regime (mc <= 5)",
        fontsize=10,
        alpha=0.7,
        ha="center",
        va="top",
    )
    # Only label the bottom-row x-axes.
    for ax in axes[2:]:
        ax.set_xlabel("max_calls")
    fig.suptitle(
        "Adaptive generate: metrics vs call budget (exact at max_calls = 8)",
        fontsize=15,
    )
    axes[-1].legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Choice config",
        fontsize=11,
        frameon=False,
    )
    _save_fig(fig, "generate_sweep")


# =============================================================================
# Tables
# =============================================================================


def _fmt(x, nd=3):
    if pd.isna(x):
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "✓" if bool(x) else ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def save_ttest_results(stats_df: pd.DataFrame) -> None:
    """Two-metric Welch t-test detail (wide, one row per variation)."""
    cols = [
        "Variation",
        "N",
        "Mean conf (correct)",
        "Mean conf (incorrect)",
        "Welch t (conf)",
        "p (conf)",
        "Cohen's d (conf)",
        "Sig conf (uncorrected)",
        "Sig conf (Bonferroni)",
        "Sig conf (BH-FDR)",
        "Mean gap (correct)",
        "Mean gap (incorrect)",
        "Welch t (gap)",
        "p (gap)",
        "Cohen's d (gap)",
        "Sig gap (uncorrected)",
        "Sig gap (Bonferroni)",
        "Sig gap (BH-FDR)",
    ]
    out = stats_df[cols].copy()
    out.to_csv(EXPERIMENT_DIR / "ttest_results.csv", index=False)

    with open(EXPERIMENT_DIR / "ttest_results.txt", "w") as f:
        f.write("One-tailed Welch t-tests: Correct vs Incorrect predictions\n")
        f.write("H1: mean(metric | Correct) > mean(metric | Incorrect)\n")
        f.write("Two metric families — confidence and score gap (top - 2nd prob.).\n")
        f.write(
            f"Multiple-comparison correction (Bonferroni, BH-FDR) applied within "
            f"each family of {len(out)} tests at alpha = {ALPHA}.\n"
        )
        f.write("Score-gap stats use the subset of items with >=2 scored labels.\n")
        f.write("=" * 78 + "\n\n")
        f.write(out.to_string(index=False))
        f.write("\n")


def _tex_table(stats_df: pd.DataFrame) -> str:
    """Paper-friendly LaTeX fragment (core columns of tab:confidence)."""
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"& \textbf{Conf.} & \textbf{Conf.} & \textbf{Corr.} & \textbf{Gap} & "
        r"\textbf{Gap} & \textbf{Welch} & \textbf{Welch} \\",
        r"\textbf{Variation} & \textbf{(correct)} & \textbf{(incorrect)} & "
        r"$\boldsymbol{r}$ & \textbf{(correct)} & \textbf{(incorrect)} & "
        r"$\boldsymbol{t}$ (conf) & $\boldsymbol{t}$ (gap) \\",
        r"\midrule",
    ]
    prev_classifier = None
    for _, row in stats_df.iterrows():
        clf = row["classifier"]
        if prev_classifier is not None and clf != prev_classifier:
            lines.append(r"\addlinespace")
        prev_classifier = clf
        lines.append(
            " & ".join(
                [
                    str(row["Variation"]).replace("&", r"\&"),
                    _fmt(row["Mean conf (correct)"], 3),
                    _fmt(row["Mean conf (incorrect)"], 3),
                    _fmt(row["Corr r"], 3),
                    _fmt(row["Mean gap (correct)"], 3),
                    _fmt(row["Mean gap (incorrect)"], 3),
                    _fmt(row["Welch t (conf)"], 2),
                    _fmt(row["Welch t (gap)"], 2),
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def save_confidence_table(stats_df: pd.DataFrame) -> None:
    """Combined paper-ready table: CSV (full) + LaTeX fragment + txt."""
    full = stats_df.copy()
    full.to_csv(EXPERIMENT_DIR / "confidence_table.csv", index=False)

    with open(EXPERIMENT_DIR / "confidence_table.txt", "w") as f:
        f.write("Confidence & calibration summary (per variation)\n")
        f.write("=" * 78 + "\n\n")
        f.write(full.to_string(index=False))
        f.write("\n")

    with open(EXPERIMENT_DIR / "confidence_table.tex", "w") as f:
        f.write("% Generated by analyze_confidence.py from results.xlsx\n")
        f.write("% Score gap = top - second-highest class probability, per item.\n")
        f.write("% Welch t (conf) / (gap): one-tailed, H1 correct > incorrect.\n\n")
        f.write(_tex_table(stats_df))
        f.write("\n")


def build_threshold_discrimination(
    long_df: pd.DataFrame, stats_df: pd.DataFrame
) -> pd.DataFrame:
    """Per-variation Youden-optimal thresholds for confidence and for the margin.

    Validates that a single threshold on either the predicted-label confidence
    or the score gap (margin) separates correct from incorrect predictions.
    Reports AUC and the Youden-optimal operating point
    (threshold, selective accuracy among items above it, coverage retained).
    """
    auc_map = {
        r["Variation"]: (r["AUC (conf)"], r["AUC (gap)"])
        for _, r in stats_df.iterrows()
    }
    rows = []
    for var in VARIATIONS:
        name = var["display"]
        sub = _variation_subset(long_df, name)
        if sub.empty:
            continue
        conf = sub["confidence"].to_numpy(dtype=float)
        gap = sub["score_gap"].to_numpy(dtype=float)
        correct = sub["correct"].to_numpy()
        c_thr, c_sa, c_cov, c_j = _youden_threshold(conf, correct)
        g_thr, g_sa, g_cov, g_j = _youden_threshold(gap, correct)
        auc_conf, auc_gap = auc_map.get(name, (float("nan"), float("nan")))
        rows.append(
            {
                "Variation": name,
                "classifier": var["classifier"],
                "method": var["method"],
                "max_calls": var["max_calls"],
                "AUC (conf)": auc_conf,
                "conf-thr (Youden)": c_thr,
                "Sel. acc (conf)": c_sa,
                "Cov. (conf)": c_cov,
                "J (conf)": c_j,
                "AUC (gap)": auc_gap,
                "gap-thr (Youden)": g_thr,
                "Sel. acc (gap)": g_sa,
                "Cov. (gap)": g_cov,
                "J (gap)": g_j,
            }
        )
    return pd.DataFrame(rows)


def _threshold_tex(td_df: pd.DataFrame) -> str:
    """Paper-friendly LaTeX fragment for tab:threshold (all variations)."""
    lines = [
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{Confidence threshold}} & "
        r"\multicolumn{4}{c}{\textbf{Margin (score-gap) threshold}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"\textbf{Variation} & \textbf{AUC} & \textbf{thr.} & \textbf{sel.\ acc.} "
        r"& \textbf{cov.} & \textbf{AUC} & \textbf{thr.} & \textbf{sel.\ acc.} "
        r"& \textbf{cov.} \\",
        r"\midrule",
    ]
    prev_classifier = None
    for _, row in td_df.iterrows():
        clf = row["classifier"]
        if prev_classifier is not None and clf != prev_classifier:
            lines.append(r"\addlinespace")
        prev_classifier = clf
        lines.append(
            " & ".join(
                [
                    str(row["Variation"]).replace("&", r"\&"),
                    _fmt(row["AUC (conf)"], 3),
                    _fmt(row["conf-thr (Youden)"], 2),
                    (
                        f'{row["Sel. acc (conf)"] * 100:.1f}\\%'
                        if pd.notna(row["Sel. acc (conf)"])
                        else ""
                    ),
                    (
                        f'{row["Cov. (conf)"]:.1f}\\%'
                        if pd.notna(row["Cov. (conf)"])
                        else ""
                    ),
                    _fmt(row["AUC (gap)"], 3),
                    _fmt(row["gap-thr (Youden)"], 2),
                    (
                        f'{row["Sel. acc (gap)"] * 100:.1f}\\%'
                        if pd.notna(row["Sel. acc (gap)"])
                        else ""
                    ),
                    (
                        f'{row["Cov. (gap)"]:.1f}\\%'
                        if pd.notna(row["Cov. (gap)"])
                        else ""
                    ),
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def save_threshold_discrimination(td_df: pd.DataFrame) -> None:
    """Threshold-discrimination table: CSV + LaTeX fragment + txt."""
    td_df.to_csv(EXPERIMENT_DIR / "threshold_discrimination.csv", index=False)
    with open(EXPERIMENT_DIR / "threshold_discrimination.txt", "w") as f:
        f.write("Threshold discrimination: confidence vs score gap (margin)\n")
        f.write("Youden-optimal threshold (J = TPR - FPR) per variation;\n")
        f.write(
            "sel. acc. = accuracy among items with score >= threshold; cov. = % retained.\n"
        )
        f.write("=" * 78 + "\n\n")
        f.write(td_df.to_string(index=False))
        f.write("\n")
    with open(EXPERIMENT_DIR / "threshold_discrimination.tex", "w") as f:
        f.write("% Generated by analyze_confidence.py from results.xlsx\n")
        f.write(
            "% thr. = Youden-optimal threshold; sel. acc. / cov. at that threshold.\n\n"
        )
        f.write(_threshold_tex(td_df))
        f.write("\n")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    df = pd.read_excel(RESULTS_XLSX, sheet_name=SHEET)
    long_df = build_long_frame(df)

    stats_df, conf_pvals, gap_pvals = compute_stats(long_df)

    plot_boxplots(
        long_df,
        value_col="confidence",
        pvals=conf_pvals,
        ylabel="Confidence",
        title="Confidence distribution: Correct vs Incorrect predictions",
        stem="confidence_boxplots",
    )
    plot_boxplots(
        long_df,
        value_col="score_gap",
        pvals=gap_pvals,
        ylabel="Score gap (top - 2nd probability)",
        title="Score-gap distribution: Correct vs Incorrect predictions",
        stem="scoregap_boxplots",
    )
    plot_faceted_threshold(
        long_df,
        which="acc",
        value_col="confidence",
        ylabel="Selective accuracy",
        title="Selective accuracy vs confidence threshold",
        stem="selective_accuracy",
        ylim=(0.0, 1.02),
        xlabel="Confidence threshold",
    )
    plot_faceted_threshold(
        long_df,
        which="cov",
        value_col="confidence",
        ylabel="Coverage (%)",
        title="Coverage vs confidence threshold",
        stem="coverage_curve",
        ylim=(0.0, 100.5),
        xlabel="Confidence threshold",
    )
    plot_faceted_threshold(
        long_df,
        which="acc",
        value_col="score_gap",
        ylabel="Selective accuracy",
        title="Selective accuracy vs score-gap (margin) threshold",
        stem="selective_accuracy_margin",
        ylim=(0.0, 1.02),
        xlabel="Score-gap threshold",
    )
    plot_faceted_threshold(
        long_df,
        which="cov",
        value_col="score_gap",
        ylabel="Coverage (%)",
        title="Coverage vs score-gap (margin) threshold",
        stem="coverage_curve_margin",
        ylim=(0.0, 100.5),
        xlabel="Score-gap threshold",
    )
    plot_calibration(long_df)
    plot_generate_sweep(stats_df)

    save_ttest_results(stats_df)
    save_confidence_table(stats_df)
    save_threshold_discrimination(build_threshold_discrimination(long_df, stats_df))

    print("t-test results (confidence):")
    print(
        stats_df[
            [
                "Variation",
                "Mean conf (correct)",
                "Mean conf (incorrect)",
                "Welch t (conf)",
                "p (conf)",
                "Cohen's d (conf)",
            ]
        ].to_string(index=False)
    )
    print("\nScore-gap t-tests:")
    print(
        stats_df[
            [
                "Variation",
                "Mean gap (correct)",
                "Mean gap (incorrect)",
                "Welch t (gap)",
                "p (gap)",
                "Cohen's d (gap)",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs written to: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
