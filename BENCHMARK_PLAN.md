# Petey Extraction Benchmark Plan

## Goal

Systematically evaluate how **parser**, **OCR backend**, and **LLM model** choices
affect extraction accuracy across two document types:

1. **Medical (ED notes)** — Claude-generated synthetic PDFs with clean embedded text
2. **PAR (DHCR decisions)** — real-world government PDFs, some with degraded/scanned pages

## Approach

Vary one axis at a time to isolate each variable's impact:

- **Parser sweep**: 6 parsers, 1 fixed model — measures how text extraction quality affects LLM accuracy
- **Model sweep**: 1 fixed parser, 6 models — measures how LLM capability affects accuracy given identical input
- **OCR sweep** (PAR only): 5 OCR backends, 1 fixed parser + model — measures OCR quality on scanned/degraded pages

## Constraints

- No GPU available — all local tools must be CPU-friendly
- API-based services are fine (Datalab, Unstructured, Textract, Google Document AI, Chandra, Surya, Mistral, Google Vision)
- Docling excluded (too slow)
- EasyOCR / PaddleOCR excluded (GPU-dependent)

## Configurations

### Medical Data — Parser Sweep (6 runs)

| # | Parser | OCR | Model |
|---|--------|-----|-------|
| 1 | pymupdf | none | gpt-4.1-mini |
| 2 | marker | none | gpt-4.1-mini |
| 3 | unstructured | none | gpt-4.1-mini |
| 4 | textract | none | gpt-4.1-mini |
| 5 | google_documentai | none | gpt-4.1-mini |

### Medical Data — Model Sweep (6 runs, 1 overlap with parser sweep)

| # | Parser | OCR | Model |
|---|--------|-----|-------|
| 1 | pymupdf | none | gpt-4.1-mini |
| 7 | pymupdf | none | gpt-4.1 |
| 8 | pymupdf | none | gpt-4o |
| 9 | pymupdf | none | claude-sonnet-4-6 |
| 10 | pymupdf | none | claude-haiku-4-5 |
| 11 | pymupdf | none | gemini/gemini-2.5-flash |

### PAR Data — Parser Sweep (6 runs)

| # | Parser | OCR | Model |
|---|--------|-----|-------|
| 12 | pymupdf | none | gpt-4.1-mini |
| 13 | marker | none | gpt-4.1-mini |
| 14 | unstructured | none | gpt-4.1-mini |
| 15 | textract | none | gpt-4.1-mini |
| 16 | google_documentai | none | gpt-4.1-mini |

### PAR Data — Model Sweep (6 runs, 1 overlap with parser sweep)

| # | Parser | OCR | Model |
|---|--------|-----|-------|
| 12 | pymupdf | none | gpt-4.1-mini |
| 18 | pymupdf | none | gpt-4.1 |
| 19 | pymupdf | none | gpt-4o |
| 20 | pymupdf | none | claude-sonnet-4-6 |
| 21 | pymupdf | none | claude-haiku-4-5 |
| 22 | pymupdf | none | gemini/gemini-2.5-flash |

### PAR Data — OCR Sweep (5 runs)

Uses pdfplumber as the parser because pymupdf runs its own internal OCR
that can't be fully disabled.

| # | Parser | OCR | Model |
|---|--------|-----|-------|
| 23 | pdfplumber | tesseract | gpt-4.1-mini |
| 24 | pdfplumber | mistral | gpt-4.1-mini |
| 25 | pdfplumber | chandra | gpt-4.1-mini |
| 26 | pdfplumber | surya | gpt-4.1-mini |
| 27 | pdfplumber | google_vision | gpt-4.1-mini |

## Totals

- **25 unique configurations** (29 total minus 4 duplicates from overlap)
- 11 medical runs, 14 PAR runs (including OCR)

## Data

All benchmark data lives in `../benchmarks/`:

```
benchmarks/
├── claude_med/                  # 102 synthetic ED note PDFs
├── claude_med_ground_truth.csv  # ground truth (name, age, sex, complaint1, complaint2, outcome)
├── claude_med_schema.yaml       # petey schema for ED notes
├── par_decision/                # 115 real-world DHCR PAR decision PDFs
├── par_ground_truth.csv         # ground truth (14 fields, pivoted from raw)
├── par_ground_truth_raw.csv     # raw long-format ground truth with pipeline comparison columns
└── par_decision_schema.yaml     # petey schema for PAR decisions
```

## Evaluation

Each run is scored against ground truth using the same evaluation pipeline:
- **Medical**: `claude_med_ground_truth.csv` (102 synthetic patients, 6 fields each)
- **PAR**: `par_ground_truth.csv` (114 PAR decisions, 14 fields each — manually verified)

Metrics: per-field similarity scores, overall mean, pass/partial/fail counts.

## Key Questions

1. **Parser impact**: Does structured output (Datalab markdown, Google/Textract layout) help the LLM vs raw text (pymupdf)?
2. **Model impact**: Is gpt-4.1 worth the cost over gpt-4.1-mini? How do Anthropic/Google models compare?
3. **OCR impact**: Which OCR backend best handles degraded PAR pages (garbled date stamps, faded text)?
4. **Cross-dataset consistency**: Do the best configs for clean medical PDFs also win on messy PAR docs?

## Auth Requirements

| Service | Auth method |
|---------|------------|
| Datalab, Chandra, Surya | `DATALAB_API_KEY` (shared key) |
| Unstructured API | `UNSTRUCTURED_API_KEY` |
| AWS Textract | AWS credentials (boto3 — `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` or IAM role) |
| Google Document AI | GCP service account JSON (`GOOGLE_APPLICATION_CREDENTIALS`) |
| Google Vision OCR | GCP service account JSON (`GOOGLE_APPLICATION_CREDENTIALS`, same as above) |
| Mistral OCR | `MISTRAL_API_KEY` |
| OpenAI models | `OPENAI_API_KEY` |
| Anthropic models | `ANTHROPIC_API_KEY` |
| Gemini (via litellm) | `GEMINI_API_KEY` |
