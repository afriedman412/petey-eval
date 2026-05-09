"""
score_results.py — Score benchmark results against ground truth
================================================================
Designed to run in Colab with GPU for sentence-transformers.

Usage (Colab):
    from google.colab import auth
    auth.authenticate_user()

    from score_results import score_file, score_all_from_gcs

    # Score a single result file
    df = score_file(
        "gs://benchmarks/results/par_simple/datalab_gpt-4.1_run1.json",
        "gs://benchmarks/data/par_simple/ground_truth.csv",
        "gs://benchmarks/data/par_simple/schema.yaml",
    )

    # Score everything in the bucket
    all_scores = score_all_from_gcs()
"""

import csv
import io
import json
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------

_embed_fn = None


def _get_embed():
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[embedder] sentence-transformers / all-MiniLM-L6-v2")

        def embed(texts):
            return model.encode(texts, normalize_embeddings=True)

        _embed_fn = embed
        return embed
    except ImportError:
        pass
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("[embedder] TF-IDF fallback (install sentence-transformers for better results)")
    _vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    _corpus = []

    def embed(texts):
        combined = list(set(_corpus + texts))
        _vec.fit(combined)
        _corpus.clear()
        _corpus.extend(combined)
        mat = _vec.transform(texts).toarray()
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms

    _embed_fn = embed
    return embed


def cosine_sim(s1, s2):
    """Cosine similarity between two strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    embed = _get_embed()
    vecs = embed([str(s1), str(s2)])
    return float(np.dot(vecs[0], vecs[1]))


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_ENUM_ALIASES = {
    "m": "male", "man": "male", "masculine": "male",
    "f": "female", "woman": "female", "feminine": "female",
    "nb": "non-binary", "nonbinary": "non-binary",
}


def _norm(val):
    """Normalize a value for comparison."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ("none", "null", "n/a"):
        return ""
    return s


def _norm_enum(val):
    """Normalize enum values."""
    s = _norm(val).lower()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    s = s.replace("_", " ")
    return _ENUM_ALIASES.get(s, s)


def _normalize_date(val):
    """Try to normalize date to YYYY-MM-DD."""
    s = _norm(val)
    if not s:
        return ""
    try:
        from dateutil import parser as dp
        return dp.parse(s).strftime("%Y-%m-%d")
    except (ValueError, OverflowError, ImportError):
        return s


# ---------------------------------------------------------------------------
# Per-field scoring
# ---------------------------------------------------------------------------

PASS_THRESH = 0.85


def score_field(gt_val, pred_val, field_name, field_type=None):
    """Score a single field. Returns (score, match).

    score: float 0-1
    match: 1 if score >= PASS_THRESH, else 0
    """
    gt = _norm(gt_val)
    pred = _norm(pred_val)

    # Both empty = match
    if not gt and not pred:
        return 1.0, 1

    # One empty = fail
    if not gt or not pred:
        return 0.0, 0

    # Date fields: normalize then exact match
    if field_type == "date" or "date" in field_name.lower():
        gt_d = _normalize_date(gt)
        pred_d = _normalize_date(pred)
        if gt_d == pred_d:
            return 1.0, 1
        return 0.0, 0

    # Enum fields: normalized exact match
    if field_type == "enum" or field_type == "category":
        if _norm_enum(gt) == _norm_enum(pred):
            return 1.0, 1
        return 0.0, 0

    # Exact match (case-insensitive)
    if gt.lower() == pred.lower():
        return 1.0, 1

    # Token containment
    gt_tokens = set(gt.lower().split())
    pred_tokens = set(pred.lower().split())
    if gt_tokens and gt_tokens.issubset(pred_tokens):
        return 1.0, 1

    # Cosine similarity
    sc = cosine_sim(gt, pred)
    return sc, 1 if sc >= PASS_THRESH else 0


# ---------------------------------------------------------------------------
# Score a result file
# ---------------------------------------------------------------------------

def score_file(results_json, gt_csv, schema_yaml=None):
    """Score a benchmark result file against ground truth.

    Args:
        results_json: path or URL to result JSON (benchmark.py format)
        gt_csv: path or URL to ground truth CSV
        schema_yaml: optional path/URL to schema YAML (for field types)

    Returns:
        DataFrame with columns:
            source_file, field1, field2, ..., fieldN
        Values are 1 (match) or 0 (no match).
        Also includes _score_field1, _score_field2, ... with raw similarity scores.
        Plus metadata columns: _dataset, _parser, _model, _errors.
    """
    # Load results
    if isinstance(results_json, str):
        data = _load_json(results_json)
    else:
        data = results_json

    if isinstance(data, dict):
        meta = data.get("meta", {})
        recs = data.get("results", [])
    else:
        meta = {}
        recs = data

    dataset = meta.get("dataset", "")
    parser = meta.get("parser", "")
    model = meta.get("model", "")

    # Load ground truth
    if isinstance(gt_csv, str) and (
        gt_csv.startswith("gs://") or gt_csv.startswith("http") or
        (len(gt_csv) < 500 and not gt_csv.startswith("source_file"))
    ):
        gt_text = _load_text(gt_csv)
    else:
        gt_text = gt_csv
    gt = {}
    reader = csv.DictReader(io.StringIO(gt_text))
    for row in reader:
        gt[row["source_file"]] = row
    fields = [k for k in list(gt.values())[0].keys() if k != "source_file"]

    # Load schema for field types
    field_types = {}
    if schema_yaml:
        import yaml
        if isinstance(schema_yaml, dict):
            spec = schema_yaml
        elif isinstance(schema_yaml, str) and (
            schema_yaml.startswith("gs://") or schema_yaml.startswith("http") or
            len(schema_yaml) < 500
        ):
            spec = yaml.safe_load(_load_text(schema_yaml))
        else:
            spec = yaml.safe_load(schema_yaml)
        for name, cfg in spec.get("fields", {}).items():
            field_types[name] = cfg.get("type")

    # Score each record
    rows = []
    for rec in recs:
        if rec.get("_error"):
            continue
        sf = rec.get("_source_file", "")
        if sf not in gt:
            continue

        row = {"source_file": sf}
        for field in fields:
            gt_val = gt[sf].get(field)
            pred_val = rec.get(field)
            ftype = field_types.get(field)
            score, match = score_field(gt_val, pred_val, field, ftype)
            row[field] = match
            row[f"_score_{field}"] = round(score, 3)

        rows.append(row)

    df = pd.DataFrame(rows)
    df["_dataset"] = dataset
    df["_parser"] = parser
    df["_model"] = model
    df["_errors"] = sum(1 for r in recs if r.get("_error"))
    return df


# ---------------------------------------------------------------------------
# File loading helpers (local or GCS)
# ---------------------------------------------------------------------------

def _load_json(path):
    """Load JSON from local path or GCS URL."""
    if path.startswith("gs://"):
        from google.cloud import storage
        bucket_name, blob_path = path[5:].split("/", 1)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
        return json.loads(blob.download_as_text())
    elif path.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(path) as resp:
            return json.loads(resp.read())
    else:
        with open(path) as f:
            return json.load(f)


def _load_text(path):
    """Load text from local path or GCS URL."""
    if path.startswith("gs://"):
        from google.cloud import storage
        bucket_name, blob_path = path[5:].split("/", 1)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
        return blob.download_as_text()
    elif path.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(path) as resp:
            return resp.read().decode()
    else:
        with open(path) as f:
            return f.read()


# ---------------------------------------------------------------------------
# Score all results from GCS
# ---------------------------------------------------------------------------

DATASETS = {
    "medical": {
        "gt": "gs://benchmarks/results/claude_med_ground_truth.csv",
        "schema": "gs://benchmarks/results/claude_med_schema.yaml",
        "results_prefix": "gs://benchmarks/results/medical/",
    },
    "par_simple": {
        "gt": "gs://benchmarks/results/par_ground_truth.csv",
        "schema": "gs://benchmarks/results/par_decision_schema_simple.yaml",
        "results_prefix": "gs://benchmarks/results/par_simple/",
    },
    "par_detailed": {
        "gt": "gs://benchmarks/results/par_ground_truth.csv",
        "schema": "gs://benchmarks/results/par_decision_schema.yaml",
        "results_prefix": "gs://benchmarks/results/par_detailed/",
    },
}


def score_all_from_gcs(datasets=None, min_clean_pct=0.5):
    """Score all result files from GCS.

    Args:
        datasets: list of dataset names, or None for all
        min_clean_pct: skip files where error rate exceeds this

    Returns:
        DataFrame with all scores concatenated.
    """
    from google.cloud import storage
    client = storage.Client()

    if datasets is None:
        datasets = list(DATASETS.keys())

    all_dfs = []
    for ds_name in datasets:
        ds = DATASETS[ds_name]
        gt_csv = _load_text(ds["gt"])
        schema = None
        try:
            import yaml
            schema = yaml.safe_load(_load_text(ds["schema"]))
        except Exception:
            pass

        # List result files
        bucket_name = "benchmarks"
        prefix = ds["results_prefix"].replace(f"gs://{bucket_name}/", "")
        blobs = list(client.bucket(bucket_name).list_blobs(prefix=prefix))

        for blob in blobs:
            fname = blob.name.split("/")[-1]
            if not fname.endswith(".json") or "run_log" in fname or fname.startswith("extraction_"):
                continue

            print(f"  Scoring {ds_name}/{fname}...", end=" ", flush=True)
            try:
                data = json.loads(blob.download_as_text())
                recs = data.get("results", data) if isinstance(
                    data, dict) else data
                errors = sum(1 for r in recs if r.get("_error"))
                total = len(recs)
                if total > 0 and errors / total > min_clean_pct:
                    print(f"SKIP ({errors}/{total} errors)")
                    continue

                df = score_file(data, gt_csv, schema)
                df["_file"] = fname
                all_dfs.append(df)
                print(f"OK ({len(df)} records)")
            except Exception as e:
                print(f"ERROR: {e}")

    if not all_dfs:
        return {}

    combined = pd.concat(all_dfs, ignore_index=True)

    # Split by dataset type
    result = {}
    med_mask = combined["_dataset"] == "medical"
    par_mask = combined["_dataset"].isin(["par_simple", "par_detailed"])

    if med_mask.any():
        df = combined[med_mask].reset_index(drop=True)
        result["medical"] = df.dropna(axis=1, how="all")
    if par_mask.any():
        df = combined[par_mask].reset_index(drop=True)
        result["par"] = df.dropna(axis=1, how="all")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(df, meta):
    """Print a per-run report: throughput, per-field match, overall accuracy."""
    field_cols = [
        c for c in df.columns
        if not c.startswith("_") and c != "source_file"
    ]
    parser = meta.get("parser", "?")
    model = meta.get("model", "?")
    dataset = meta.get("dataset", "?")
    timing = meta.get("timing") or {}

    print()
    print("=" * 64)
    print(f"  {dataset} × {parser} × {model}")
    print("=" * 64)
    if timing:
        bs = timing.get("batch_seconds")
        fps = timing.get("files_per_second")
        n = timing.get("file_count")
        if bs and n:
            per_min = round(60 * n / bs, 1) if bs else None
            print(f"  Throughput: {fps} files/s  "
                  f"({per_min} files/min, {n} files in {bs}s)")
    print(f"  Records scored: {len(df)}")
    if len(df) > 0:
        print(f"  Errors excluded: {int(df['_errors'].iloc[0])}")

    if field_cols and len(df) > 0:
        print("\n  Per-field match rate:")
        for f in field_cols:
            print(f"    {f:32s} {df[f].mean():.3f}")
        print(f"\n  Overall accuracy: {df[field_cols].mean().mean():.3f}")
    print()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Score a single benchmark result JSON against ground "
                    "truth and print a report."
    )
    p.add_argument("results_json",
                   help="Path to result JSON from benchmark.py")
    p.add_argument("ground_truth_csv",
                   help="Path to ground truth CSV")
    p.add_argument("schema_yaml",
                   help="Path to schema YAML")
    p.add_argument("--out",
                   help="Optional path to save the per-doc scored CSV")
    args = p.parse_args()

    with open(args.results_json) as _f:
        _data = json.load(_f)
    _meta = _data.get("meta", {}) if isinstance(_data, dict) else {}

    df = score_file(_data, args.ground_truth_csv, args.schema_yaml)
    _print_report(df, _meta)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"  Wrote: {args.out}\n")
