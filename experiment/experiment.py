#!/usr/bin/env python3
"""
experiment.py — Zero-shot classification benchmark:
  BART (facebook/bart-large-mnli) vs ollama-classifier (Qwen2.5 3B-Instruct, v0.5.0)
  vs scikit-llm (same Ollama model via OpenAI-compatible endpoint)

24 variations (numbered in run order; see ``variations.py`` for the catalog):

  BART (NLI zero-shot baseline):
    1.  BART — subclass names only
    2.  BART — subclass names only + opt-out

  ollama-classifier `classify` (v0.5.0, method="multi_call":
        multi-call geometric-mean completion scoring, always exact):
    3.  classify — subclass names only
    4.  classify — subclass names only + opt-out
    5.  classify — subclass names + descriptions
    6.  classify — subclass names + descriptions + opt-out

  scikit-llm (ZeroShotGPTClassifier, single-call label-only):
    7. scikit-llm — subclass names only
    8. scikit-llm — subclass names only + opt-out

  ollama-classifier `generate` (v0.5.0, method="adaptive_generate":
        adaptive trie-masked generation with divergence-aware confidence;
        4 choice-configs × 4 call budgets = 16):
    9-12.   generate names only          [max_calls ∈ {1, 3, 5, 8}]
    13-16.  generate names only + opt-out [max_calls ∈ {1, 3, 5, 8}]
    17-20.  generate names + descriptions [max_calls ∈ {1, 3, 5, 8}]
    21-24.  generate desc. + opt-out      [max_calls ∈ {1, 3, 5, 8}]

Outputs:
  results.xlsx — Detailed Results (per-variation prediction + confidence +
                 probabilities +, for ollama-classifier, method / approximate /
                 coverage), Summary, per-class metrics, confidence detail,
                 and confidence summary sheets
  summary.txt  — summary statistics
"""

import json
import time
import warnings
from pathlib import Path

import pandas as pd
from scipy import stats
from tqdm import tqdm

from variations import VARIATION_SPECS

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct"

BART_MODEL = "facebook/bart-large-mnli"

# scikit-llm drives the same Ollama backend/model through its OpenAI-compatible
# endpoint. scikit-llm 0.4.x has no native custom-URL support and hardcodes the
# OpenAI base URL, so the redirect is applied at runtime in classify_skllm().
SKLLM_MODEL = OLLAMA_MODEL
OLLAMA_OPENAI_URL = OLLAMA_HOST.rstrip("/") + "/v1"

DATASET_URL = (
    "https://zenodo.org/records/18459651/files/manual_labels_coicop2018.csv"
    "?download=1"
)
DATASET_PATH = Path(__file__).parent / "manual_labels_coicop2018.csv"
OUTPUT_XLSX = Path(__file__).parent / "results.xlsx"
OUTPUT_SUMMARY = Path(__file__).parent / "summary.txt"

COICOP_SUBCLASSES = {
    "01.2.1.0": {
        "name": "Fruit and vegetable juices",
        "description": "Includes: * fruit and vegetable juices, unfermented and not containing added alcohol, whether or not they contain added sugar or other sweetening matter; * syrups of fruit and vegetables and concentrates of fruit and vegetables; * powdered juices",
    },
    "01.2.2.0": {
        "name": "Coffee and coffee substitutes",
        "description": "Includes: * coffee, whether or not decaffeinated, roasted or ground, including instant coffee; * coffee substitutes; * extracts, essences and concentrates of coffee; * coffee-based beverage preparations",
    },
    "01.2.3.0": {
        "name": "Tea, maté and other plant products for infusion",
        "description": "Includes: * green tea, unfermented; black tea, fermented; and tea, maté and other plant products, partly fermented, for infusion; * tea substitutes and tea extracts and essences; * fruit and herbal teas; * rooibos tea; * instant tea; * iced tea",
    },
    "01.2.4.0": {
        "name": "Cocoa drinks",
        "description": "Includes: * cocoa and chocolate-based drinks",
    },
    "01.2.5.0": {
        "name": "Water",
        "description": "Includes: Mineral and spring waters, still or sparkling, without added ingredients.",
    },
    "01.2.6.0": {
        "name": "Soft drinks",
        "description": "Includes: * soft drinks, such as sodas, lemonades and colas; * sparkling juices",
    },
    "01.2.9.0": {
        "name": "Other non-alcoholic beverages",
        "description": "Includes: * flavoured water; * energy drinks; energy supplements (which are mixed with water for drink preparation); and protein powder (mixed with water for drink preparation); * birch juice and sap, and aloe vera juice and drinks; * syrups and concentrates for the preparation of beverages; * other non-alcoholic beverages",
    },
}

OPT_OUT_LABEL = "None of the above"


# =============================================================================
# Dataset
# =============================================================================


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        print("Downloading dataset from Zenodo...")
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)
    df["code"] = df["code"].astype(str)
    df = df[df["code"].str.startswith("01.2")].copy()
    df = df[df["code"].isin(COICOP_SUBCLASSES.keys())].copy()
    df = df.reset_index(drop=True)

    print(f"Dataset: {len(df)} products across {df['code'].nunique()} subclasses")
    print("  Subclass distribution:")
    for code, count in df["code"].value_counts().sort_index().items():
        print(f"    {code} ({COICOP_SUBCLASSES[code]['name']}): {count}")
    return df


# =============================================================================
# BART zero-shot classifier
# =============================================================================


def classify_bart(
    texts: list[str],
    candidate_labels: list[str],
    multi_label: bool = False,
) -> list[dict]:
    from transformers import pipeline

    classifier = pipeline(
        "zero-shot-classification",
        model=BART_MODEL,
        device=-1,
    )

    results = []
    for text in tqdm(texts, desc="BART classification", unit="item"):
        output = classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=multi_label,
        )
        top_label = output["labels"][0]
        top_score = output["scores"][0]
        all_probs = dict(zip(output["labels"], output["scores"]))
        results.append(
            {
                "label": top_label,
                "confidence": top_score,
                "probabilities": all_probs,
            }
        )
    return results


# =============================================================================
# ollama-classifier (v0.5.0 API: LLMClassifier + OllamaBackend)
# =============================================================================

_classifier = None


def get_classifier():
    """Lazily build and reuse a single ``LLMClassifier`` across all variations.

    In v0.5.0 the entry point is
    ``LLMClassifier(OllamaBackend(model, host), max_workers=N)``: the backend wraps
    the Ollama inference engine, and the classifier holds a thread pool used by
    the batch/sync helpers. Constructing once avoids re-creating the backend and
    thread pool per call, so wall-clock timing reflects pure inference cost.
    """
    global _classifier
    if _classifier is None:
        from ollama_classifier import LLMClassifier
        from ollama_classifier.backends import OllamaBackend

        backend = OllamaBackend(model=OLLAMA_MODEL, host=OLLAMA_HOST)
        _classifier = LLMClassifier(backend, max_workers=8)
    return _classifier


def classify_ollama_method(
    texts: list[str],
    choices: dict[str, str] | list[str],
    method: str,
    max_calls: int | None = None,
) -> list[dict]:
    """Run one Ollama classification strategy over ``texts``.

    In v0.5.0 both ``classify`` and ``generate`` return a full
    ``ClassificationResult`` carrying ``prediction``, ``confidence``,
    ``probabilities``, ``method``, ``approximate``, ``coverage`` and ``n_calls``.

    - ``classify`` scores every label as a prompt completion without generation,
      applying geometric-mean normalization (always exact). It reports
      ``method="multi_call"``, ``approximate=False``, ``coverage={}``, and makes
      N calls for N labels.
    - ``generate`` performs adaptive trie-masked constrained generation with
      divergence-aware confidence; it reports ``method="adaptive_generate"``,
      makes 1 to ``max_calls`` calls, and sets ``approximate=True`` when any label
      has partial token coverage (``coverage < 1.0``). ``coverage`` maps each
      label to the fraction of its tokens that were scored.

    ``choices`` is the final set of candidate labels (opt-out already appended
    by the caller when applicable).
    """
    classifier = get_classifier()

    if method == "generate" and max_calls is not None:
        tag = f"generate, max_calls={max_calls}"
    else:
        tag = method

    results = []
    for text in tqdm(texts, desc=f"Ollama ({tag})", unit="item"):
        try:
            if method == "classify":
                result = classifier.classify(text, choices)
            elif method == "generate":
                result = classifier.generate(text, choices, max_calls=max_calls)
            else:
                raise ValueError(f"Unknown method: {method}")

            results.append(
                {
                    "label": result.prediction,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
                    "method": result.method,
                    "approximate": result.approximate,
                    "coverage": result.coverage,
                    "n_calls": result.n_calls,
                }
            )
        except Exception as e:
            print(f"  ERROR on '{text[:50]}...': {e}")
            results.append(
                {
                    "label": "ERROR",
                    "confidence": None,
                    "probabilities": {},
                    "method": method,
                    "approximate": None,
                    "coverage": {},
                    "n_calls": 0,
                }
            )

    return results


# =============================================================================
# scikit-llm (ZeroShotGPTClassifier) — same Ollama backend via OpenAI-compat API
# =============================================================================


def classify_skllm(
    texts: list[str],
    candidate_labels: list[str],
    use_opt_out: bool = False,
) -> list[dict]:
    import openai
    import skllm.openai.chatgpt as _skllm_chatgpt
    from skllm.config import SKLLMConfig
    from skllm.models.gpt.gpt_zero_shot_clf import ZeroShotGPTClassifier

    # scikit-llm 0.4.x hardcodes openai.api_base to the OpenAI endpoint inside
    # set_credentials(); redirect it to Ollama's OpenAI-compatible endpoint.
    def _ollama_credentials(key: str, org: str) -> None:
        openai.api_key = key
        openai.organization = org
        openai.api_type = "open_ai"
        openai.api_version = None
        openai.api_base = OLLAMA_OPENAI_URL

    _skllm_chatgpt.set_credentials = _ollama_credentials
    SKLLMConfig.set_openai_key("ollama")  # dummy key; Ollama ignores it

    default_label = OPT_OUT_LABEL if use_opt_out else "ERROR"
    labels = list(candidate_labels) + ([OPT_OUT_LABEL] if use_opt_out else [])

    clf = ZeroShotGPTClassifier(
        openai_model=SKLLM_MODEL,
        default_label=default_label,
    )
    clf.fit(texts, labels)

    preds = clf.predict(texts)
    return [{"label": str(p), "confidence": None, "probabilities": {}} for p in preds]


# =============================================================================
# Summary statistics
# =============================================================================


def compute_summary(
    df_results: pd.DataFrame,
    ground_truth_col: str,
    prediction_col: str,
    has_opt_out: bool,
    label: str,
    elapsed_seconds: float = None,
) -> dict:
    n_total = len(df_results)
    gt = df_results[ground_truth_col]
    pred = df_results[prediction_col]

    correct_overall = (pred == gt).sum()
    pct_correct_overall = correct_overall / n_total * 100

    if has_opt_out:
        classified_mask = ~pred.isin([OPT_OUT_LABEL, "ERROR"])
    else:
        classified_mask = pred != "ERROR"

    n_classified = classified_mask.sum()
    pct_classified = n_classified / n_total * 100

    if n_classified > 0:
        correct_on_classified = (pred[classified_mask] == gt[classified_mask]).sum()
        pct_correct_on_classified = correct_on_classified / n_classified * 100
    else:
        pct_correct_on_classified = 0.0

    n_opt_out = (pred == OPT_OUT_LABEL).sum() if has_opt_out else 0
    n_errors = (pred == "ERROR").sum()

    macro = compute_macro_metrics(gt, pred, has_opt_out)

    return {
        "Variation": label,
        "N": n_total,
        "Accuracy (%)": round(pct_correct_overall, 1),
        "Precision (macro)": round(macro["precision"], 3),
        "Recall (macro)": round(macro["recall"], 3),
        "F1 (macro)": round(macro["f1"], 3),
        "% Classified": round(pct_classified, 1) if has_opt_out else "—",
        "% Correct (on classified)": (
            round(pct_correct_on_classified, 1) if has_opt_out else "—"
        ),
        "N Opt-out": n_opt_out if has_opt_out else "—",
        "N Errors": n_errors,
        "Time (s)": round(elapsed_seconds, 1) if elapsed_seconds else "—",
    }


def compute_per_class_metrics(
    gt: pd.Series,
    pred: pd.Series,
    has_opt_out: bool,
    class_names: dict[str, str],
) -> pd.DataFrame:
    if has_opt_out:
        mask = ~pred.isin([OPT_OUT_LABEL, "ERROR"])
    else:
        mask = pred != "ERROR"

    gt_filtered = gt[mask]
    pred_filtered = pred[mask]
    classes = sorted(gt_filtered.unique())

    rows = []
    for cls in classes:
        tp = ((pred_filtered == cls) & (gt_filtered == cls)).sum()
        fp = ((pred_filtered == cls) & (gt_filtered != cls)).sum()
        fn = ((pred_filtered != cls) & (gt_filtered == cls)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        support = (gt_filtered == cls).sum()

        rows.append(
            {
                "Class": cls,
                "Name": class_names.get(cls, cls),
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1": round(f1, 3),
                "Support": support,
            }
        )

    return pd.DataFrame(rows)


def compute_macro_metrics(
    gt: pd.Series,
    pred: pd.Series,
    has_opt_out: bool,
) -> dict:
    if has_opt_out:
        mask = ~pred.isin([OPT_OUT_LABEL, "ERROR"])
    else:
        mask = pred != "ERROR"

    gt_filtered = gt[mask]
    pred_filtered = pred[mask]
    classes = sorted(gt_filtered.unique())

    precisions, recalls, f1s = [], [], []
    for cls in classes:
        tp = ((pred_filtered == cls) & (gt_filtered == cls)).sum()
        fp = ((pred_filtered == cls) & (gt_filtered != cls)).sum()
        fn = ((pred_filtered != cls) & (gt_filtered == cls)).sum()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    n = len(classes)
    return {
        "precision": sum(precisions) / n if n > 0 else 0.0,
        "recall": sum(recalls) / n if n > 0 else 0.0,
        "f1": sum(f1s) / n if n > 0 else 0.0,
    }


# =============================================================================
# Confidence analysis
# =============================================================================


def _score_gap_from_probs(probs: dict) -> float | None:
    """Margin of victory between the top and second-highest class probabilities.

    Standard uncertainty/margin metric: the gap between the most probable label
    and its nearest competitor. Returns None when fewer than two scored
    candidates are present (the margin is undefined).
    """
    if not isinstance(probs, dict) or len(probs) < 2:
        return None
    values = sorted((float(v) for v in probs.values()), reverse=True)
    return round(values[0] - values[1], 4)


def compute_confidence_analysis(
    gt: pd.Series,
    pred: pd.Series,
    confidences: pd.Series,
    probs_json: pd.Series,
    has_opt_out: bool,
) -> dict:
    if has_opt_out:
        valid_mask = ~pred.isin([OPT_OUT_LABEL, "ERROR"])
    else:
        valid_mask = pred != "ERROR"

    gt_v = gt[valid_mask]
    pred_v = pred[valid_mask]
    conf_v = confidences[valid_mask]
    probs_v = probs_json[valid_mask]

    correct_mask = pred_v == gt_v
    incorrect_mask = ~correct_mask

    n_correct = correct_mask.sum()
    n_incorrect = incorrect_mask.sum()

    if n_correct == 0 and n_incorrect == 0:
        return {
            "N Correct": 0,
            "N Incorrect": 0,
            "Mean Conf (Correct)": None,
            "Mean Conf (Incorrect)": None,
            "Corr. Coefficient": None,
            "Corr. p-value": None,
            "Mean Score Gap": None,
            "Median Score Gap": None,
        }

    mean_conf_correct = float(conf_v[correct_mask].mean()) if n_correct > 0 else None
    mean_conf_incorrect = (
        float(conf_v[incorrect_mask].mean()) if n_incorrect > 0 else None
    )

    is_correct = correct_mask.astype(float)
    valid_conf = conf_v.dropna()
    valid_binary = is_correct.loc[valid_conf.index]

    if len(valid_conf) >= 3 and valid_binary.nunique() > 1:
        corr, pvalue = stats.pointbiserialr(valid_binary, valid_conf)
        corr_coeff = round(corr, 4)
        corr_pval = round(pvalue, 6)
    else:
        corr_coeff = None
        corr_pval = None

    # Score gap = margin of victory (top probability minus second-highest),
    # evaluated over incorrect predictions only.
    gaps = []
    for idx in incorrect_mask.index:
        probs = json.loads(probs_v.loc[idx]) if pd.notna(probs_v.loc[idx]) else {}
        gap = _score_gap_from_probs(probs)
        if gap is not None:
            gaps.append(gap)

    if gaps:
        mean_gap = round(float(sum(gaps) / len(gaps)), 4)
        median_gap = round(float(sorted(gaps)[len(gaps) // 2]), 4)
    else:
        mean_gap = None
        median_gap = None

    return {
        "Variation": "",
        "N Correct": n_correct,
        "N Incorrect": n_incorrect,
        "Mean Conf (Correct)": (
            round(mean_conf_correct, 4) if mean_conf_correct is not None else None
        ),
        "Mean Conf (Incorrect)": (
            round(mean_conf_incorrect, 4) if mean_conf_incorrect is not None else None
        ),
        "Corr. Coefficient": corr_coeff,
        "Corr. p-value": corr_pval,
        "Mean Score Gap": mean_gap,
        "Median Score Gap": median_gap,
    }


def build_confidence_detail(
    texts: pd.Series,
    gt: pd.Series,
    pred: pd.Series,
    confidences: pd.Series,
    probs_json: pd.Series,
    has_opt_out: bool,
    name_to_code: dict[str, str],
) -> pd.DataFrame:
    if has_opt_out:
        valid_mask = ~pred.isin([OPT_OUT_LABEL, "ERROR"])
    else:
        valid_mask = pred != "ERROR"

    gt_v = gt[valid_mask]
    pred_v = pred[valid_mask]
    conf_v = confidences[valid_mask]
    probs_v = probs_json[valid_mask]
    texts_v = texts[valid_mask]

    rows = []
    for idx in gt_v.index:
        gt_code = gt_v.loc[idx]
        pred_code = pred_v.loc[idx]
        gt_name = name_to_code.get(gt_code, gt_code)
        conf = float(conf_v.loc[idx]) if pd.notna(conf_v.loc[idx]) else None
        probs = json.loads(probs_v.loc[idx]) if pd.notna(probs_v.loc[idx]) else {}
        prob_correct = float(probs.get(gt_name, 0.0))
        is_correct = pred_code == gt_code
        score_gap = _score_gap_from_probs(probs)

        rows.append(
            {
                "text": texts_v.loc[idx],
                "ground_truth": gt_code,
                "predicted": pred_code,
                "correct": is_correct,
                "confidence": round(conf, 4) if conf is not None else None,
                "prob_correct_label": round(prob_correct, 4),
                "score_gap": round(score_gap, 4) if score_gap is not None else None,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main experiment
# =============================================================================


def main():
    print("=" * 70)
    print("Zero-shot classification benchmark")
    print(f"  Ollama: {OLLAMA_MODEL} @ {OLLAMA_HOST}")
    print(f"  BART:   {BART_MODEL}")
    print("=" * 70)

    df = load_dataset()
    texts = df["name"].tolist()

    names_only = [info["name"] for info in COICOP_SUBCLASSES.values()]
    names_with_desc = {
        info["name"]: info["description"] for info in COICOP_SUBCLASSES.values()
    }

    code_to_name = {code: info["name"] for code, info in COICOP_SUBCLASSES.items()}

    results_df = df[["name", "category", "code"]].copy()
    results_df.rename(columns={"code": "ground_truth"}, inplace=True)

    summary_rows = []
    per_class_dfs = {}
    conf_analyses = {}
    conf_detail_dfs = {}
    key_display = {}

    name_to_code = {v: k for k, v in code_to_name.items()}

    def _with_opt_out(choices):
        if isinstance(choices, dict):
            return {**choices, OPT_OUT_LABEL: "Use if no other option fits"}
        return list(choices) + [OPT_OUT_LABEL]

    # Variations are defined once in ``variations.py`` (VARIATION_SPECS) so this
    # writer and the reader in ``analyze_confidence.py`` share a single source
    # of truth for keys and run order. Here we attach the data-dependent choice
    # sets and the display name to each spec.
    _flavor_choices = {
        "names_only": names_only,
        "names_with_desc": names_with_desc,
    }

    variations = []
    for spec in VARIATION_SPECS:
        base_choices = _flavor_choices[spec.flavor]
        if spec.classifier == "skllm":
            # scikit-llm always gets the bare names list; opt-out is passed as a
            # flag and appended inside classify_skllm.
            choices = names_only
            use_opt_out = spec.has_opt_out
        else:  # bart / ollama — fold opt-out into the candidate set
            choices = _with_opt_out(base_choices) if spec.has_opt_out else base_choices
            use_opt_out = False

        name = spec.base_display()
        if spec.method == "generate":
            name = f"{name} [generate, max_calls={spec.max_calls}]"

        var = {
            "name": name,
            "key": spec.key,
            "has_opt_out": spec.has_opt_out,
            "classifier": spec.classifier,
            "choices": choices,
            "use_opt_out": use_opt_out,
        }
        if spec.method:
            var["method"] = spec.method
        if spec.max_calls is not None:
            var["max_calls"] = spec.max_calls
        variations.append(var)

    for var in variations:
        print(f"\n--- {var['name']} ---")
        t0 = time.time()

        if var["classifier"] == "bart":
            results = classify_bart(texts, var["choices"])
        elif var["classifier"] == "skllm":
            results = classify_skllm(
                texts,
                var["choices"],
                use_opt_out=var.get("use_opt_out", False),
            )
        else:  # ollama (classify or generate)
            results = classify_ollama_method(
                texts,
                var["choices"],
                method=var["method"],
                max_calls=var.get("max_calls"),
            )

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        col_key = var["key"]
        key_display[col_key] = var["name"]
        results_df[col_key] = [r["label"] for r in results]
        results_df[f"{col_key}_conf"] = [r["confidence"] for r in results]
        results_df[f"{col_key}_code"] = results_df[col_key].map(
            lambda x: name_to_code.get(x, x)
        )
        results_df[f"{col_key}_probs"] = [
            json.dumps(r.get("probabilities", {})) for r in results
        ]
        results_df[f"{col_key}_ncalls"] = [r.get("n_calls") for r in results]
        # ollama-classifier v0.5.0 metadata (BART / scikit-llm omit these keys,
        # so .get() yields None / {} — no false signal for those classifiers).
        results_df[f"{col_key}_method"] = [r.get("method") for r in results]
        results_df[f"{col_key}_approx"] = [r.get("approximate") for r in results]
        results_df[f"{col_key}_coverage"] = [
            json.dumps(r.get("coverage", {})) for r in results
        ]

        summary_row = compute_summary(
            results_df,
            "ground_truth",
            f"{col_key}_code",
            has_opt_out=var["has_opt_out"],
            label=var["name"],
            elapsed_seconds=elapsed,
        )
        ncalls_col = results_df[f"{col_key}_ncalls"]
        summary_row["Total N calls"] = (
            int(ncalls_col.fillna(0).sum()) if ncalls_col.notna().any() else "—"
        )
        # Count items flagged approximate (partial token coverage) by the
        # adaptive_generate strategy. classify / BART / scikit-llm carry no
        # such flag, so they report "—".
        approx_col = results_df[f"{col_key}_approx"]
        summary_row["N Approximate"] = (
            int(approx_col.fillna(False).astype(bool).sum())
            if approx_col.notna().any()
            else "—"
        )
        summary_rows.append(summary_row)

        per_class_dfs[col_key] = compute_per_class_metrics(
            results_df["ground_truth"],
            results_df[f"{col_key}_code"],
            has_opt_out=var["has_opt_out"],
            class_names=code_to_name,
        )

        conf_analysis = compute_confidence_analysis(
            results_df["ground_truth"],
            results_df[f"{col_key}_code"],
            results_df[f"{col_key}_conf"],
            results_df[f"{col_key}_probs"],
            has_opt_out=var["has_opt_out"],
        )
        conf_analysis["Variation"] = var["name"]
        conf_analyses[col_key] = conf_analysis

        conf_detail_dfs[col_key] = build_confidence_detail(
            results_df["name"],
            results_df["ground_truth"],
            results_df[f"{col_key}_code"],
            results_df[f"{col_key}_conf"],
            results_df[f"{col_key}_probs"],
            has_opt_out=var["has_opt_out"],
            name_to_code=name_to_code,
        )

    # =========================================================================
    # Save results
    # =========================================================================
    summary_df = pd.DataFrame(summary_rows)

    results_df["ground_truth_name"] = results_df["ground_truth"].map(code_to_name)

    conf_summary_df = pd.DataFrame(conf_analyses.values())

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        conf_summary_df.to_excel(writer, sheet_name="Confidence Summary", index=False)
        for col_key, pc_df in per_class_dfs.items():
            sheet_name = f"PC_{col_key[:28]}"
            pc_df.to_excel(writer, sheet_name=sheet_name, index=False)
        for col_key, cd_df in conf_detail_dfs.items():
            sheet_name = f"CD_{col_key[:28]}"
            cd_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {OUTPUT_XLSX}")
    print(f"{'=' * 70}")

    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    print("\nConfidence Analysis:")
    print(conf_summary_df.to_string(index=False))

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write("Zero-shot Classification Benchmark — Summary Statistics\n")
        f.write(f"Ollama model: {OLLAMA_MODEL} @ {OLLAMA_HOST}\n")
        f.write(f"BART model:   {BART_MODEL}\n")
        f.write(
            f"Dataset:      {len(df)} products, {df['code'].nunique()} subclasses\n"
        )
        f.write("=" * 70 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        for col_key, pc_df in per_class_dfs.items():
            f.write(f"\n--- Per-class metrics: {key_display[col_key]} ---\n")
            f.write(pc_df.to_string(index=False))
            f.write("\n")
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("Confidence Analysis\n")
        f.write("=" * 70 + "\n\n")
        f.write(conf_summary_df.to_string(index=False))
        f.write("\n")


if __name__ == "__main__":
    main()
