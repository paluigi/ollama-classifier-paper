# Plan: Confidence-analysis plots & t-tests for the benchmark

## Goal
Create a new, standalone analysis script that reads
`experiment/results.xlsx` (produced by `experiment/experiment.py`) and produces
4 figures + 1 t-test result table, saving all outputs in `experiment/`.

## Confirmed decisions (from user)
- **Library:** `matplotlib` + `seaborn` (add via `uv add matplotlib seaborn`).
- **Granularity:** each variation is its own "methodology".
- **scikit-llm:** excluded from all confidence-based outputs (no `_conf` data).
- **Output formats:** PNG (300 DPI) **and EPS** per figure.

## Scope: the 6 confidence-bearing variations
| Display name | col_key | opt-out |
|---|---|---|
| BART (names only) | `bart_names_only` | no |
| BART (names + opt-out) | `bart_names_+_opt-out` | yes |
| Ollama (names only) | `ollama_names_only` | no |
| Ollama (names + opt-out) | `ollama_names_+_opt-out` | yes |
| Ollama (names + descriptions) | `ollama_names_+_descriptions` | no |
| Ollama (desc + opt-out) | `ollama_desc_+_opt-out` | yes |

scikit-llm variations (`scikit-llm_names_only`, `scikit-llm_names+opt-out`) are
**dropped** entirely.

## Stated assumptions (correct me if wrong)
1. **Opt-out / ERROR items are excluded** from every confidence analysis
   (mirrors the existing `compute_confidence_analysis` logic in
   `experiment.py`). For opt-out variations, predictions equal to
   `"None of the above"` or `"ERROR"` are dropped before any plot/test.
   → correct/incorrect is judged only on actually-classified items.
2. **correct** = `{col_key}_code == ground_truth`; **incorrect** otherwise
   (on the valid/classified subset).
3. **Threshold grid** for plots 3 & 4: `np.linspace(0, 1, 101)` (0.00–1.00, step 0.01).
4. **Plot 3 "accuracy rate"**: among items with `_conf >= threshold`, the fraction
   correct. Plot stops the line once the surviving subset is empty.
5. **Plot 4 "percentage of products"**: `count(conf >= threshold) / N_valid * 100`,
   where `N_valid` = number of classified items for that variation (its own denominator).
6. **t-test**: one-tailed **Welch's** t-test, H1 = mean(conf|incorrect) < mean(conf|correct),
   i.e. `scipy.stats.ttest_ind(correct, incorrect, alternative="greater", equal_var=False)`.

## File to create
`experiment/analyze_confidence.py` — flat procedural script (matches the style of
the sibling `experiment.py`; kept in `experiment/` rather than a new `src/` for
consistency with the existing experiment layout). Type hints + docstrings; no OOP
justified for a one-off plotting script. Runnable via `uv run python experiment/analyze_confidence.py`.

## Script structure
1. **Config**: paths (`results.xlsx`, output dir), variation metadata table
   (display name, col_key, has_opt_out), threshold grid.
2. **Load + reshape**:
   - Read "Detailed Results" sheet.
   - For each variation build a long DataFrame:
     `methodology, confidence, correct (bool)`, after excluding opt-out/ERROR and NaN confidence.
3. **Task 1 — Boxplots**: `seaborn.FacetGrid` (or `catplot`) `col=methodology`,
   6 panels (3x2). Within each panel: `x=correct (Correct/Incorrect)`,
   `y=confidence`, boxplot + strip/swarm optional overlay. Annotate each panel
   with its one-tailed p-value. Save `confidence_boxplots.png` + `.eps`.
4. **Task 2 — t-tests**: loop over variations; collect
   N_correct, N_incorrect, mean_correct, mean_incorrect, Welch t-stat, one-tailed p.
   Save `ttest_results.csv` (and a human-readable `ttest_results.txt`).
5. **Task 3 — Selective accuracy vs threshold**: per variation, for each threshold
   compute accuracy of `conf >= threshold` subset; plot one line per variation.
   x=confidence threshold, y=accuracy. Save `selective_accuracy.png` + `.eps`.
6. **Task 4 — Coverage vs threshold**: per variation, % of valid items with
   `conf >= threshold`; one line per variation. Save `coverage_curve.png` + `.eps`.
7. **Common plotting style**: shared legend, distinct palette per variation,
   labeled axes/titles, tight_layout, `dpi=300` for PNG.

## Outputs (all in `experiment/`)
- `confidence_boxplots.png` / `confidence_boxplots.eps`
- `selective_accuracy.png` / `selective_accuracy.eps`
- `coverage_curve.png` / `coverage_curve.eps`
- `ttest_results.csv`
- `ttest_results.txt`

## Dependency change
`uv add matplotlib seaborn` (updates `pyproject.toml` + `uv.lock`).

## Validation
- Run `uv run python experiment/analyze_confidence.py` and confirm 5 output files
  are created and figures render legibly (6 methodologies, no scikit-llm).
- Spot-check t-test p-values against the existing Confidence Summary in
  `summary.txt` (mean-conf direction should match: correct > incorrect).
- Run `ruff check` + `ruff format` on the new file.
