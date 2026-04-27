# Experiment: BART vs ollama-classifier Benchmark

Zero-shot classification comparison on COICOP 2018 Division 01.2 (Non-alcoholic beverages).

## Setup

```bash
pip install pandas openpyxl transformers torch ollama ollama-classifier tqdm
```

## Configuration

Edit the following constants at the top of `experiment.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server endpoint on your LAN | `http://192.168.178.XX:11434` |
| `OLLAMA_MODEL` | Model name for ollama-classifier | `qwen2.1:1.5b` |
| `BART_MODEL` | HuggingFace model for BART baseline | `facebook/bart-large-mnli` |

## Run

```bash
cd experiment
python experiment.py
```

## Six Variations

| # | Method | Labels | Opt-out |
|---|--------|--------|---------|
| 1 | BART | Names only | No |
| 2 | BART | Names only | Yes |
| 3 | ollama-classifier | Names only | No |
| 4 | ollama-classifier | Names only | Yes |
| 5 | ollama-classifier | Names + descriptions | No |
| 6 | ollama-classifier | Names + descriptions | Yes |

## Outputs

- **`results.xlsx`** — Two sheets:
  - *Detailed Results*: per-product predictions with ground truth for all 6 variations
  - *Summary*: aggregate statistics (% correct, % classified, % correct on classified)
- **`summary.txt`** — Human-readable summary statistics

## Dataset

Products are filtered from the COICOP 2018 manually labeled dataset
([Zenodo](https://zenodo.org/records/18459651)), selecting only codes starting
with `01.2` (Non-alcoholic beverages). This yields ~637 products across 7 subclasses.
