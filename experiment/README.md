# Experiment: BART vs ollama-classifier vs scikit-llm Benchmark

Zero-shot classification comparison on COICOP 2018 Division 01.2 (Non-alcoholic beverages).

## Setup

```bash
uv sync
```

Dependencies are declared in the project's `pyproject.toml` and managed with
`uv`. Requires **ollama-classifier ≥ 0.5.0** (the `LLMClassifier` +
`OllamaBackend` API; `classify`/`generate` both return a `ClassificationResult`
with `prediction`, `confidence`, `probabilities`, `coverage`, and `n_calls`).

## Configuration

Edit the following constants at the top of `experiment.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server endpoint on your LAN | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Model name for ollama-classifier | `qwen2.5:3b-instruct` |
| `BART_MODEL` | HuggingFace model for BART baseline | `facebook/bart-large-mnli` |
| `SKLLM_MODEL` | Model for scikit-llm (reuses `OLLAMA_MODEL`) | `qwen2.5:3b-instruct` |

scikit-llm drives the same Ollama backend/model through Ollama's
OpenAI-compatible endpoint (`OLLAMA_HOST/v1`); no separate model server is
required. scikit-llm is a label-only classifier, so its variations report no
confidence/probability scores.

The `MAX_CALLS = [1, 3, 5, 8]` list controls the call-budget sweep for the
`generate` (adaptive) strategy.

## Run

```bash
uv run python experiment/experiment.py
uv run python experiment/analyze_confidence.py
```

## Variations

The benchmark runs **24 variations**. Each variation writes a column group
(`{key}`, `{key}_conf`, `{key}_code`, `{key}_probs`, `{key}_ncalls`) to the
*Detailed Results* sheet.

| # | Method | Strategy | Labels | Opt-out | Key |
|---|--------|----------|--------|---------|-----|
| 1 | BART | — | Names only | No | `bart_names_only` |
| 2 | BART | — | Names only | Yes | `bart_names_optout` |
| 3 | ollama-classifier | `classify` | Names only | No | `ollama_names_only` |
| 4 | ollama-classifier | `classify` | Names only | Yes | `ollama_names_optout` |
| 5 | ollama-classifier | `classify` | Names + descriptions | No | `ollama_names_desc` |
| 6 | ollama-classifier | `classify` | Names + descriptions | Yes | `ollama_desc_optout` |
| 7 | scikit-llm | — | Names only | No | `skllm_names_only` |
| 8 | scikit-llm | — | Names only | Yes | `skllm_names_optout` |
| 9–24 | ollama-classifier | `generate` (max_calls ∈ {1,3,5,8}) | each of the 4 Ollama label configs | as per config | `{base}_gen_mc{N}` |

For the `generate` strategy, the same four Ollama choice configurations used by
`classify` (names only / +opt-out / +descriptions / desc.+opt-out) are each run
at four call budgets (`max_calls` = 1, 3, 5, 8), yielding 16 adaptive-generation
variations. Since v0.5.0 `generate()` returns a full `ClassificationResult`
(label + confidence + probabilities + `coverage` + `n_calls`), so these variations
participate in the confidence analysis.

## Outputs

### `experiment.py`
- **`results.xlsx`** — sheets:
  - *Detailed Results*: per-product predictions, confidence, mapped codes,
    probability distributions, and `n_calls` for all 24 variations
  - *Summary*: aggregate metrics (accuracy, macro precision/recall/F1,
    % classified, wall-clock time, `Total N calls`)
  - *Confidence Summary*: per-variation confidence/correctness diagnostics
  - *Per-class* (`PC_{key}`) and *Confidence Detail* (`CD_{key}`) sheets per
    variation
- **`summary.txt`** — human-readable summary statistics

### `analyze_confidence.py`
Reads the *Detailed Results* sheet and produces (all in `experiment/`):
- **`confidence_boxplots.{png,eps}`** — confidence by Correct/Incorrect, faceted
  across all 22 confidence-bearing variations (BART ×2, classify ×4, generate ×16)
- **`selective_accuracy.{png,eps}`** — accuracy vs confidence threshold
- **`coverage_curve.{png,eps}`** — coverage vs confidence threshold
- **`ttest_results.{csv,txt}`** — one-tailed Welch t-tests per methodology

scikit-llm is excluded from the confidence analysis (no confidence scores).

## Dataset

Products are filtered from the COICOP 2018 manually labeled dataset
([Zenodo](https://zenodo.org/records/18459651)), selecting only codes starting
with `01.2` (Non-alcoholic beverages). This yields ~637 products across 7 subclasses.

## Change Log

- **LLMClassifier API**: Replaced the legacy `OllamaClassifier(client, model)`
  with `LLMClassifier(OllamaBackend(model, host))`, built once as a singleton so
  that per-variation timing reflects inference cost only.
- **v0.5.0 `generate()`**: Since v0.5.0, `generate()` returns a full
  `ClassificationResult` (label + confidence + probabilities + `coverage` +
  `n_calls`; previously label-only), so `generate` now participates in the
  confidence analysis. Added 16 adaptive-generation variations (4 Ollama choice
  configs × `max_calls` ∈ {1,3,5,8}).
- **v0.5.0 metadata capture**: Added per-variation `{key}_method`,
  `{key}_approx`, and `{key}_coverage` columns to *Detailed Results*, plus a
  per-variation `N Approximate` summary field (items with partial token
  coverage under the `generate` strategy).
- **Shared variation catalog**: Variation keys, run order, and the
  `generate` call-budget sweep now live in `variations.py`, imported by both
  `experiment.py` and `analyze_confidence.py`, so the writer and reader can no
  longer drift apart silently.
- **Explicit variation keys**: Column groups are keyed by stable `key` strings
  instead of name-mangling, preventing collisions among the generate budget
  variants.
- **Call accounting**: Added per-item `{key}_ncalls` columns and a per-variation
  `Total N calls` summary field to compare realized calls against the budget.
- **Confidence analysis expanded**: `analyze_confidence.py` now covers all 22
  confidence-bearing variations (was 6); boxplot grid is dynamic, palette sized
  to the variation count, and both scripts tolerate partial `results.xlsx`
  files (missing columns are skipped).
- **Lint/format**: Resolved pre-existing ruff issues; both scripts pass
  `ruff check` and `black`.
