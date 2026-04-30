#!/usr/bin/env python3
"""
experiment.py — Zero-shot classification benchmark:
  BART (facebook/bart-large-mnli) vs ollama-classifier (Qwen2.5 3B-Instruct)

Six variations:
  1. BART — subclass names only
  2. BART — subclass names only + opt-out
  3. ollama-classifier — subclass names only
  4. ollama-classifier — subclass names only + opt-out
  5. ollama-classifier — subclass names + descriptions
  6. ollama-classifier — subclass names + descriptions + opt-out

Outputs:
  results.xlsx — full predictions with ground truth per variation,
                per-class metrics, confidence detail, and confidence summary
  summary.txt   — summary statistics
"""

import json
import time
import warnings
from pathlib import Path

import pandas as pd
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct"

BART_MODEL = "facebook/bart-large-mnli"

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
        print(f"Downloading dataset from Zenodo...")
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)
    df["code"] = df["code"].astype(str)
    df = df[df["code"].str.startswith("01.2")].copy()
    df = df[df["code"].isin(COICOP_SUBCLASSES.keys())].copy()
    df = df.reset_index(drop=True)

    print(f"Dataset: {len(df)} products across {df['code'].nunique()} subclasses")
    print(f"  Subclass distribution:")
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
        results.append({
            "label": top_label,
            "confidence": top_score,
            "probabilities": all_probs,
        })
    return results


# =============================================================================
# ollama-classifier
# =============================================================================

def classify_ollama(
    texts: list[str],
    choices: dict[str, str] | list[str],
    use_opt_out: bool = False,
    method: str = "classify",
) -> list[dict]:
    from ollama import Client
    from ollama_classifier import OllamaClassifier

    client = Client(host=OLLAMA_HOST)
    classifier = OllamaClassifier(client=client, model=OLLAMA_MODEL)

    if use_opt_out:
        if isinstance(choices, dict):
            choices = {**choices, OPT_OUT_LABEL: "Use if no other option fits"}
        else:
            choices = list(choices) + [OPT_OUT_LABEL]

    results = []
    for text in tqdm(texts, desc=f"Ollama ({method})", unit="item"):
        try:
            if method == "classify":
                result = classifier.classify(text, choices)
                scores = (
                    result.scores if hasattr(result, "scores") else {}
                )
                results.append({
                    "label": result.prediction,
                    "confidence": result.confidence,
                    "probabilities": scores,
                })
            elif method == "generate":
                label = classifier.generate(text, choices)
                results.append({
                    "label": label,
                    "confidence": None,
                    "probabilities": {},
                })
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            print(f"  ERROR on '{text[:50]}...': {e}")
            results.append({"label": "ERROR", "confidence": None, "probabilities": {}})

    return results


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
        "% Correct (on classified)": round(pct_correct_on_classified, 1) if has_opt_out else "—",
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
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = (gt_filtered == cls).sum()

        rows.append({
            "Class": cls,
            "Name": class_names.get(cls, cls),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
            "Support": support,
        })

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

def compute_confidence_analysis(
    gt: pd.Series,
    pred: pd.Series,
    confidences: pd.Series,
    probs_json: pd.Series,
    has_opt_out: bool,
    name_to_code: dict[str, str],
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
    mean_conf_incorrect = float(conf_v[incorrect_mask].mean()) if n_incorrect > 0 else None

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

    gaps = []
    for idx in incorrect_mask.index:
        gt_code = gt_v.loc[idx]
        gt_name = name_to_code.get(gt_code, gt_code)
        probs = json.loads(probs_v.loc[idx]) if pd.notna(probs_v.loc[idx]) else {}
        conf_pred = float(conf_v.loc[idx]) if pd.notna(conf_v.loc[idx]) else 0.0
        conf_correct = float(probs.get(gt_name, 0.0))
        gaps.append(conf_pred - conf_correct)

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
        "Mean Conf (Correct)": round(mean_conf_correct, 4) if mean_conf_correct is not None else None,
        "Mean Conf (Incorrect)": round(mean_conf_incorrect, 4) if mean_conf_incorrect is not None else None,
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
        score_gap = (conf - prob_correct) if (conf is not None) else None

        rows.append({
            "text": texts_v.loc[idx],
            "ground_truth": gt_code,
            "predicted": pred_code,
            "correct": is_correct,
            "confidence": round(conf, 4) if conf is not None else None,
            "prob_correct_label": round(prob_correct, 4),
            "score_gap": round(score_gap, 4) if score_gap is not None else None,
        })

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
    ground_truth = df["code"].tolist()

    names_only = [info["name"] for info in COICOP_SUBCLASSES.values()]
    names_with_desc = {
        info["name"]: info["description"]
        for info in COICOP_SUBCLASSES.values()
    }

    code_to_name = {
        code: info["name"] for code, info in COICOP_SUBCLASSES.items()
    }
    gt_names = [code_to_name[c] for c in ground_truth]

    results_df = df[["name", "category", "code"]].copy()
    results_df.rename(columns={"code": "ground_truth"}, inplace=True)

    summary_rows = []
    per_class_dfs = {}
    conf_analyses = {}
    conf_detail_dfs = {}

    name_to_code = {v: k for k, v in code_to_name.items()}

    variations = [
        {
            "name": "BART (names only)",
            "has_opt_out": False,
            "classifier": "bart",
            "choices": names_only,
        },
        {
            "name": "BART (names + opt-out)",
            "has_opt_out": True,
            "classifier": "bart",
            "choices": names_only + [OPT_OUT_LABEL],
        },
        {
            "name": "Ollama (names only)",
            "has_opt_out": False,
            "classifier": "ollama",
            "choices": names_only,
            "method": "classify",
        },
        {
            "name": "Ollama (names + opt-out)",
            "has_opt_out": True,
            "classifier": "ollama",
            "choices": names_only,
            "method": "classify",
            "use_opt_out": True,
        },
        {
            "name": "Ollama (names + descriptions)",
            "has_opt_out": False,
            "classifier": "ollama",
            "choices": names_with_desc,
            "method": "classify",
        },
        {
            "name": "Ollama (desc + opt-out)",
            "has_opt_out": True,
            "classifier": "ollama",
            "choices": names_with_desc,
            "method": "classify",
            "use_opt_out": True,
        },
    ]

    for var in variations:
        print(f"\n--- {var['name']} ---")
        t0 = time.time()

        if var["classifier"] == "bart":
            results = classify_bart(texts, var["choices"])
        else:
            results = classify_ollama(
                texts,
                var["choices"],
                use_opt_out=var.get("use_opt_out", False),
                method=var.get("method", "classify"),
            )

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        col_key = var["name"].replace(" ", "_").replace("(", "").replace(")", "").lower()
        results_df[col_key] = [r["label"] for r in results]
        results_df[f"{col_key}_conf"] = [r["confidence"] for r in results]
        results_df[f"{col_key}_code"] = results_df[col_key].map(
            lambda x: name_to_code.get(x, x)
        )
        results_df[f"{col_key}_probs"] = [
            json.dumps(r.get("probabilities", {})) for r in results
        ]

        summary_rows.append(
            compute_summary(
                results_df, "ground_truth", f"{col_key}_code",
                has_opt_out=var["has_opt_out"],
                label=var["name"],
                elapsed_seconds=elapsed,
            )
        )

        per_class_dfs[var["name"]] = compute_per_class_metrics(
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
            name_to_code=name_to_code,
        )
        conf_analysis["Variation"] = var["name"]
        conf_analyses[var["name"]] = conf_analysis

        conf_detail_dfs[var["name"]] = build_confidence_detail(
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
        for var_name, pc_df in per_class_dfs.items():
            sheet_name = f"PC_{var_name[:27]}"
            pc_df.to_excel(writer, sheet_name=sheet_name, index=False)
        for var_name, cd_df in conf_detail_dfs.items():
            sheet_name = f"CD_{var_name[:27]}"
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
        f.write(f"Dataset:      {len(df)} products, {df['code'].nunique()} subclasses\n")
        f.write("=" * 70 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        for var_name, pc_df in per_class_dfs.items():
            f.write(f"\n--- Per-class metrics: {var_name} ---\n")
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
