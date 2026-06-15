# Experiment: BART vs ollama-classifier vs scikit-llm Benchmark

Zero-shot classification comparison on COICOP 2018 Division 01.2 (Non-alcoholic beverages).

## Setup

```bash
uv add pandas openpyxl transformers torch ollama ollama-classifier scikit-llm tqdm
```

## Configuration

Edit the following constants at the top of `experiment.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server endpoint on your LAN | `http://192.168.178.XX:11434` |
| `OLLAMA_MODEL` | Model name for ollama-classifier | `qwen2.5:3b-instruct` |
| `BART_MODEL` | HuggingFace model for BART baseline | `facebook/bart-large-mnli` |
| `SKLLM_MODEL` | Model for scikit-llm (reuses `OLLAMA_MODEL`) | `qwen2.5:3b-instruct` |

scikit-llm drives the same Ollama backend/model through Ollama's
OpenAI-compatible endpoint (`OLLAMA_HOST/v1`); no separate model server is
required. scikit-llm is a label-only classifier, so its variations report no
confidence/probability scores.

## Run

```bash
cd experiment
python experiment.py
```

## Variations

| # | Method | Labels | Opt-out |
|---|--------|--------|---------|
| 1 | BART | Names only | No |
| 2 | BART | Names only | Yes |
| 3 | ollama-classifier | Names only | No |
| 4 | ollama-classifier | Names only | Yes |
| 5 | ollama-classifier | Names + descriptions | No |
| 6 | ollama-classifier | Names + descriptions | Yes |
| 7 | scikit-llm | Names only | No |
| 8 | scikit-llm | Names only | Yes |

## Outputs

- **`results.xlsx`** — Two sheets:
  - *Detailed Results*: per-product predictions with ground truth for all 6 variations
  - *Summary*: aggregate statistics (% correct, % classified, % correct on classified)
- **`summary.txt`** — Human-readable summary statistics

## Dataset

Products are filtered from the COICOP 2018 manually labeled dataset
([Zenodo](https://zenodo.org/records/18459651)), selecting only codes starting
with `01.2` (Non-alcoholic beverages). This yields ~637 products across 7 subclasses.
