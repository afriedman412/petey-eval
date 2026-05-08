# petey-eval

Benchmark suite for [Petey](https://github.com/afriedman412/petey) — tests LLM and parser configurations against ground-truth datasets.

## Results

Pre-computed results are in `results/`. To explore interactively, open `charts_v2.ipynb`. To export chart PNGs, run `python export_charts.py`.

## Datasets

| Dataset | Docs | Pages/doc | Description |
|---------|------|-----------|-------------|
| **Medical** | 102 | 1 | Synthetic ED notes (clean embedded text) |
| **PAR Simple** | 114 | ~3 | DHCR administrative decisions (scanned/OCR'd), minimal schema |
| **PAR Detailed** | 114 | ~3 | Same PDFs as PAR Simple, with a detailed schema including examples and instructions |

Schemas and ground truth are in `data/`. PDFs are hosted on GCS and downloaded automatically on first run.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys
```

## Running benchmarks

```bash
# See what would run
python benchmark.py --dry-run

# Quick test — one model, one parser, one dataset, 5 docs
python benchmark.py --models gpt-4.1-mini --parsers pymupdf --datasets medical --limit 5

# Full model sweep on datalab
python benchmark.py --models gpt-4.1-mini gpt-4.1 gpt-5 claude-sonnet-4-6 --parsers datalab --datasets medical par_simple

# Upload results to GCS
python benchmark.py --gcs --datasets medical par_simple par_detailed
```

### Models

GPT-4.1 Mini, GPT-4.1, GPT-5 Mini, GPT-5, GPT-5.4, Claude Sonnet, Claude Haiku, Gemini 2.5 Flash, DeepSeek Chat

### Parsers

- **PyMuPDF** — local, free, uses Tesseract for OCR
- **Datalab** — API, $4/1K pages, best accuracy on scanned docs
- **Unstructured** — API, $3/1K pages

## Scoring

```bash
python score_results.py
```

Scores extraction results against ground truth using cosine similarity (sentence-transformers) with field-type-aware matching (exact match for dates/enums, semantic similarity for text fields). Falls back to TF-IDF if sentence-transformers is not installed, but results will be less accurate.

**Note:** Scoring with sentence-transformers benefits significantly from a GPU. For large result sets, consider running on Colab or a GPU instance.

Output is a pair of CSVs (`med_results.csv`, `par_results.csv`) with per-file, per-field match scores.

## Charts

```bash
python export_charts.py
```

Generates 8 PNGs in `charts/`:

1. Dataset stats
2. Model performance (Datalab parser)
3a. Parser comparison — Medical
3b. Parser comparison — PAR
4. Simple vs detailed schema
5. Per-field accuracy
6. OCR comparison (Tesseract vs Datalab)
7. LLM cost per 1,000 pages
8. Cost vs accuracy scatter

## Cloud Run

To run benchmarks on Cloud Run Jobs:

```bash
make deploy   # build image + create job
make run      # execute with default args
```

See `Makefile` for details.
