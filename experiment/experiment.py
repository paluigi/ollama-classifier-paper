#!/usr/bin/env python3
"""
experiment.py — Zero-shot classification benchmark:
  BART (facebook/bart-large-mnli) vs ollama-classifier (Qwen 2.1 1.5B)

Six variations:
  1. BART — subclass names only
  2. BART — subclass names only + opt-out
  3. ollama-classifier — subclass names only
  4. ollama-classifier — subclass names only + opt-out
  5. ollama-classifier — subclass names + descriptions
  6. ollama-classifier — subclass names + descriptions + opt-out

Outputs:
  results.xlsx — full predictions with ground truth per variation
  summary.txt   — summary statistics
"""

import time
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

# Ollama server endpoint (change to match your LAN)
OLLAMA_HOST = "http://192.168.178.XX:11434"
OLLAMA_MODEL = "qwen2.1:1.5b"

BART_MODEL = "facebook/bart-large-mnli"

DATASET_URL = (
    "https://zenodo.org/records/18459651/files/manual_labels_coicop2018.csv"
    "?download=1"
)
DATASET_PATH = Path(__file__).parent / "manual_labels_coicop2018.csv"
OUTPUT_XLSX = Path(__file__).parent / "results.xlsx"
OUTPUT_SUMMARY = Path(__file__).parent / "summary.txt"

# COICOP 2018 — Division 01.2 (Non-alcoholic beverages)
COICOP_SUBCLASSES = {
    "01.2.1.0": {
        "name": "Fruit and vegetable juices",
        "description": "PLACEHOLDER: Include/exclude description for 01.2.1.0",
    },
    "01.2.2.0": {
        "name": "Coffee and coffee substitutes",
        "description": "PLACEHOLDER: Include/exclude description for 01.2.2.0",
    },
    "01.2.3.0": {
        "name": "Tea, maté and other plant products for infusion",
        "description": "PLACEHOLDER: Include/exclude description for 01.2.3.0",
    },
    "01.2.4.0": {
        "name": "Cocoa drinks",
        "description": "PLACEHOLDER: Include/exclude description for 01.2.4.0",
    },
    "01.2.5.0": {
        "name": "Water",
        "description": "PLACEHOLDER: Include/exclude description for 01.2.5.0",
    },
    "01.2.6.0": {
        "name": "Soft drinks",
        "description": "PLACEHOLDER: Include/exclude description for 01.2.6.0",
    },
    "01.2.9.0": {
        "name": "Other non-alcoholic beverages",
        "description": "PLACEHOLDER: Include/exclude description for 01.2.9.0",
    },
}

OPT_OUT_LABEL = "None of the above"


# =============================================================================
# Dataset
# =============================================================================

def load_dataset() -> pd.DataFrame:
    """Download (if needed) and filter dataset to COICOP 01.2.* subclasses."""
    if not DATASET_PATH.exists():
        print(f"Downloading dataset from Zenodo...")
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)
    df["code"] = df["code"].astype(str)

    # Filter to 01.2.* subclasses
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
    """
    Zero-shot classification using BART-large-MNLI via HuggingFace pipeline.

    Returns list of dicts with 'label' and 'score' (confidence).
    """
    from transformers import pipeline

    classifier = pipeline(
        "zero-shot-classification",
        model=BART_MODEL,
        device=-1,  # CPU
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
    method: str = "generate",
) -> list[dict]:
    """
    Classification using ollama-classifier library.

    Args:
        texts: Product descriptions to classify.
        choices: Either a list of label names, or a dict mapping
                 label names to descriptions.
        use_opt_out: If True, adds an opt-out option.
        method: "generate" (fastest, no confidence) or
                "classify" (with confidence scores).

    Returns list of dicts with 'label' and 'confidence'.
    """
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
            if method == "generate":
                label = classifier.generate(text, choices)
                results.append({
                    "label": label,
                    "confidence": None,
                })
            elif method == "classify":
                result = classifier.classify(text, choices)
                results.append({
                    "label": result.prediction,
                    "confidence": result.confidence,
                })
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            print(f"  ERROR on '{text[:50]}...': {e}")
            results.append({"label": "ERROR", "confidence": None})

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
) -> dict:
    """
    Compute summary statistics for one variation.

    Returns dict with:
      - label: variation name
      - n_total: total number of products
      - pct_correct_overall: % correct out of total
      - pct_classified: % of products that received a label (not opt-out / error)
      - pct_correct_on_classified: % correct among classified items only
      - n_opt_out: number of opt-outs
      - n_errors: number of errors
    """
    n_total = len(df_results)
    gt = df_results[ground_truth_col]
    pred = df_results[prediction_col]

    # Overall correctness (opt-outs and errors count as incorrect)
    correct_overall = (pred == gt).sum()
    pct_correct_overall = correct_overall / n_total * 100

    # Products that received an actual label (not opt-out, not error)
    if has_opt_out:
        classified_mask = ~pred.isin([OPT_OUT_LABEL, "ERROR"])
    else:
        classified_mask = pred != "ERROR"

    n_classified = classified_mask.sum()
    pct_classified = n_classified / n_total * 100

    # Correct among classified only
    if n_classified > 0:
        correct_on_classified = (pred[classified_mask] == gt[classified_mask]).sum()
        pct_correct_on_classified = correct_on_classified / n_classified * 100
    else:
        pct_correct_on_classified = 0.0

    n_opt_out = (pred == OPT_OUT_LABEL).sum() if has_opt_out else 0
    n_errors = (pred == "ERROR").sum()

    return {
        "Variation": label,
        "N": n_total,
        "% Correct (overall)": round(pct_correct_overall, 1),
        "% Classified": round(pct_classified, 1) if has_opt_out else "—",
        "% Correct (on classified)": round(pct_correct_on_classified, 1) if has_opt_out else "—",
        "N Opt-out": n_opt_out if has_opt_out else "—",
        "N Errors": n_errors,
    }


# =============================================================================
# Main experiment
# =============================================================================

def main():
    print("=" * 70)
    print("Zero-shot classification benchmark")
    print(f"  Ollama: {OLLAMA_MODEL} @ {OLLAMA_HOST}")
    print(f"  BART:   {BART_MODEL}")
    print("=" * 70)

    # --- Load data ---
    df = load_dataset()
    texts = df["name"].tolist()
    ground_truth = df["code"].tolist()

    # --- Prepare label sets ---
    names_only = [info["name"] for info in COICOP_SUBCLASSES.values()]
    names_with_desc = {
        info["name"]: info["description"]
        for info in COICOP_SUBCLASSES.values()
    }

    # Map COICOP codes to subclass names (for BART comparison)
    code_to_name = {
        code: info["name"] for code, info in COICOP_SUBCLASSES.items()
    }
    gt_names = [code_to_name[c] for c in ground_truth]

    # --- Results storage ---
    results_df = df[["name", "category", "code"]].copy()
    results_df.rename(columns={"code": "ground_truth"}, inplace=True)

    summary_rows = []

    # =========================================================================
    # Variation 1: BART — names only
    # =========================================================================
    print("\n--- Variation 1: BART (names only) ---")
    t0 = time.time()
    bart_results_1 = classify_bart(texts, names_only)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    col = "bart_names"
    results_df[col] = [r["label"] for r in bart_results_1]
    results_df[f"{col}_conf"] = [r["confidence"] for r in bart_results_1]

    # Convert BART predictions to COICOP codes for accuracy check
    name_to_code = {v: k for k, v in code_to_name.items()}
    results_df[f"{col}_code"] = results_df[col].map(name_to_code)

    summary_rows.append(
        compute_summary(
            results_df, "ground_truth", f"{col}_code",
            has_opt_out=False,
            label="BART (names only)",
        )
    )

    # =========================================================================
    # Variation 2: BART — names only + opt-out
    # =========================================================================
    print("\n--- Variation 2: BART (names only + opt-out) ---")
    t0 = time.time()
    bart_results_2 = classify_bart(texts, names_only + [OPT_OUT_LABEL])
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    col = "bart_names_optout"
    results_df[col] = [r["label"] for r in bart_results_2]
    results_df[f"{col}_conf"] = [r["confidence"] for r in bart_results_2]
    results_df[f"{col}_code"] = results_df[col].map(
        lambda x: name_to_code.get(x, x)
    )

    summary_rows.append(
        compute_summary(
            results_df, "ground_truth", f"{col}_code",
            has_opt_out=True,
            label="BART (names + opt-out)",
        )
    )

    # =========================================================================
    # Variation 3: ollama-classifier — names only (generate)
    # =========================================================================
    print("\n--- Variation 3: Ollama (names only) ---")
    t0 = time.time()
    ollama_results_3 = classify_ollama(
        texts, names_only, use_opt_out=False, method="generate"
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    col = "ollama_names"
    results_df[col] = [r["label"] for r in ollama_results_3]
    results_df[f"{col}_conf"] = [r["confidence"] for r in ollama_results_3]
    results_df[f"{col}_code"] = results_df[col].map(
        lambda x: name_to_code.get(x, x)
    )

    summary_rows.append(
        compute_summary(
            results_df, "ground_truth", f"{col}_code",
            has_opt_out=False,
            label="Ollama (names only)",
        )
    )

    # =========================================================================
    # Variation 4: ollama-classifier — names only + opt-out (generate)
    # =========================================================================
    print("\n--- Variation 4: Ollama (names only + opt-out) ---")
    t0 = time.time()
    ollama_results_4 = classify_ollama(
        texts, names_only, use_opt_out=True, method="generate"
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    col = "ollama_names_optout"
    results_df[col] = [r["label"] for r in ollama_results_4]
    results_df[f"{col}_conf"] = [r["confidence"] for r in ollama_results_4]
    results_df[f"{col}_code"] = results_df[col].map(
        lambda x: name_to_code.get(x, x)
    )

    summary_rows.append(
        compute_summary(
            results_df, "ground_truth", f"{col}_code",
            has_opt_out=True,
            label="Ollama (names + opt-out)",
        )
    )

    # =========================================================================
    # Variation 5: ollama-classifier — names + descriptions (generate)
    # =========================================================================
    print("\n--- Variation 5: Ollama (names + descriptions) ---")
    t0 = time.time()
    ollama_results_5 = classify_ollama(
        texts, names_with_desc, use_opt_out=False, method="generate"
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    col = "ollama_desc"
    results_df[col] = [r["label"] for r in ollama_results_5]
    results_df[f"{col}_conf"] = [r["confidence"] for r in ollama_results_5]
    results_df[f"{col}_code"] = results_df[col].map(
        lambda x: name_to_code.get(x, x)
    )

    summary_rows.append(
        compute_summary(
            results_df, "ground_truth", f"{col}_code",
            has_opt_out=False,
            label="Ollama (names + descriptions)",
        )
    )

    # =========================================================================
    # Variation 6: ollama-classifier — names + descriptions + opt-out (generate)
    # =========================================================================
    print("\n--- Variation 6: Ollama (names + descriptions + opt-out) ---")
    t0 = time.time()
    ollama_results_6 = classify_ollama(
        texts, names_with_desc, use_opt_out=True, method="generate"
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    col = "ollama_desc_optout"
    results_df[col] = [r["label"] for r in ollama_results_6]
    results_df[f"{col}_conf"] = [r["confidence"] for r in ollama_results_6]
    results_df[f"{col}_code"] = results_df[col].map(
        lambda x: name_to_code.get(x, x)
    )

    summary_rows.append(
        compute_summary(
            results_df, "ground_truth", f"{col}_code",
            has_opt_out=True,
            label="Ollama (desc + opt-out)",
        )
    )

    # =========================================================================
    # Save results
    # =========================================================================
    summary_df = pd.DataFrame(summary_rows)

    # Add ground truth subclass names for readability
    results_df["ground_truth_name"] = results_df["ground_truth"].map(code_to_name)

    # Save to Excel with two sheets
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {OUTPUT_XLSX}")
    print(f"{'=' * 70}")

    # Print summary table
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    # Save summary as text too
    with open(OUTPUT_SUMMARY, "w") as f:
        f.write("Zero-shot Classification Benchmark — Summary Statistics\n")
        f.write(f"Ollama model: {OLLAMA_MODEL} @ {OLLAMA_HOST}\n")
        f.write(f"BART model:   {BART_MODEL}\n")
        f.write(f"Dataset:      {len(df)} products, {df['code'].nunique()} subclasses\n")
        f.write("=" * 70 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")


if __name__ == "__main__":
    main()
