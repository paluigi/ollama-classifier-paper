# Plan: Add scikit-llm variation to experiment.py

## Goal
Add **scikit-llm** (`ZeroShotGPTClassifier`) as a third classifier in
`experiment/experiment.py`, driving the **same Ollama backend and model**
(`qwen2.5:3b-instruct` @ `http://127.0.0.1:11434`) used by `ollama-classifier`,
so the comparison isolates prompt/scoring differences. Add **2 variations**
(names-only, names+opt-out), mirroring the two BART variations. All outputs
(`results.xlsx`, `summary.txt`) update automatically via the existing
variations-driven loop.

## Confirmed design (from research + user answers)
- **Backend**: scikit-llm talks OpenAI-compatible APIs. Point it at Ollama's
  `/v1` endpoint via `SKLLMConfig.set_gpt_url("http://127.0.0.1:11434/v1")`,
  model string `custom_url::qwen2.5:3b-instruct` (parsed by scikit-llm's
  `split_to_api_and_model` → api=`custom_url`, model=`qwen2.5:3b-instruct`).
- **Auth**: Ollama needs no key, but the `openai` client requires a non-None
  `api_key`; set a dummy `SKLLMConfig.set_openai_key("ollama")`.
- **Confidence**: scikit-llm returns **labels only** (no scores) → confidence
  columns = `None`, probabilities = `{}`. Existing confidence-analysis code
  already handles `None` (dropna / null rows), so no logic changes needed.
- **Invalid outputs (user choice: sentinel)**: set `ZeroShotGPTClassifier`
  `default_label`:
  - names-only → `default_label="ERROR"` (garbage → ERROR, counts as wrong).
  - names+opt-out → `default_label=OPT_OUT_LABEL` (garbage → opt-out).
  - This avoids scikit-llm's non-deterministic `"Random"` default.
- **Labels**: candidate classes are the subclass **names** (same `names_only`
  list BART uses); `name_to_code` mapping in the existing loop converts names→codes.
- **Determinism**: scikit-llm forces `temperature=0.0` for non-`gpt-o` models.

## Changes

### 1. Dependencies — `pyproject.toml`
- `uv add scikit-llm` (transitively installs `openai`). Record in deps.

### 2. `experiment/experiment.py` — configuration block (after BART_MODEL)
- Add `SKLLM_MODEL = f"custom_url::{OLLAMA_MODEL}"`  → `custom_url::qwen2.5:3b-instruct`.
- Add `OLLAMA_OPENAI_URL = OLLAMA_HOST.rstrip("/") + "/v1"`.

### 3. `experiment/experiment.py` — new `classify_skllm(...)` function
Mirror the signature/return shape of `classify_bart` / `classify_ollama`
(`list[dict]` with keys `label`, `confidence`, `probabilities`).
```python
def classify_skllm(
    texts: list[str],
    candidate_labels: list[str],
    use_opt_out: bool = False,
) -> list[dict]:
    from skllm.config import SKLLMConfig
    from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

    SKLLMConfig.set_gpt_url(OLLAMA_OPENAI_URL)
    SKLLMConfig.set_openai_key("ollama")  # dummy; Ollama ignores it

    default_label = OPT_OUT_LABEL if use_opt_out else "ERROR"
    labels = list(candidate_labels) + ([OPT_OUT_LABEL] if use_opt_out else [])

    clf = ZeroShotGPTClassifier(model=SKLLM_MODEL, default_label=default_label)
    clf.fit(texts, labels)   # registers classes_; no actual training

    preds = clf.predict(texts)   # tqdm progress handled internally
    return [
        {"label": str(p), "confidence": None, "probabilities": {}}
        for p in preds
    ]
```
- Wrap the whole body in the same `try/except` per-item error pattern used
  elsewhere is **not** required (scikit-llm batches internally); if `predict`
  raises, let it propagate (fail-fast per global dev-phase policy). The
  `default_label` sentinel already converts bad per-item outputs to
  `ERROR`/opt-out.

### 4. `experiment/experiment.py` — add 2 entries to `variations` list
```python
{
    "name": "scikit-llm (names only)",
    "has_opt_out": False,
    "classifier": "skllm",
    "choices": names_only,
},
{
    "name": "scikit-llm (names+opt-out)",
    "has_opt_out": True,
    "classifier": "skllm",
    "choices": names_only,
    "use_opt_out": True,
},
```
- Names chosen compact so `var_name[:27]` sheet-name logic stays <31 chars
  (verified: `PC_scikit-llm (names+opt-out)` = 30 chars; no truncation).

### 5. `experiment/experiment.py` — dispatch in the main loop
Extend the existing `if var["classifier"] == "bart": ... else: classify_ollama`
into an explicit chain:
```python
if var["classifier"] == "bart":
    results = classify_bart(texts, var["choices"])
elif var["classifier"] == "skllm":
    results = classify_skllm(
        texts, var["choices"], use_opt_out=var.get("use_opt_out", False)
    )
else:
    results = classify_ollama(
        texts, var["choices"],
        use_opt_out=var.get("use_opt_out", False),
        method=var.get("method", "classify"),
    )
```
- Everything downstream (`col_key` columns, `compute_summary`,
  `compute_per_class_metrics`, `compute_confidence_analysis`,
  `build_confidence_detail`, xlsx writer, summary.txt) is generic and keys off
  `var["name"]`/`col_key`, so it absorbs the new variants with **no further
  edits**.

### 6. `experiment/README.md` — doc update
- Title → "BART vs ollama-classifier vs scikit-llm".
- Setup `pip`/`uv` line: add `scikit-llm`.
- Config table: add `SKLLM_MODEL` row (note it reuses `OLLAMA_MODEL`).
- Variations table: **8 variations** (add rows 7–8 for scikit-llm).
- Note: scikit-llm rows have no confidence data (label-only classifier).

## Output impact (automatic, no extra code)
- `results.xlsx`:
  - *Detailed Results*: +4 columns × 2 variants (`...label`, `_conf`, `_code`, `_probs`).
  - *Summary*: +2 rows.
  - Per-class sheets: +2 (`PC_scikit-llm (names only)`, `PC_scikit-llm (names+opt-out)`).
  - Confidence-detail sheets: +2 (rows with null confidence/prob_correct).
  - *Confidence Summary*: +2 rows (null correlation/gaps for skllm).
- `summary.txt`: +2 summary rows + 2 per-class sections.

## Verification
1. `uv sync` after `uv add scikit-llm`.
2. `uv run ruff check experiment/experiment.py` and `uv run black --check`.
3. `uv run python experiment/experiment.py` (requires Ollama running + the
   `qwen2.5:3b-instruct` model pulled) → confirm `results.xlsx` has 8
   variations in Summary and the 2 new sheets exist.
4. Spot-check a few scikit-llm predictions map (via `name_to_code`) to codes.

## Out of scope
- No "descriptions" variant for scikit-llm (label-only method; would need a
  custom prompt template with label-mapping risks — deliberately excluded).
- No changes to BART or ollama-classifier logic.
- No tests added (only on explicit request).
