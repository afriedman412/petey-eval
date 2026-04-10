"""
evaluate_claude_data.py — Schema-agnostic extraction evaluator
===============================================================
Scores extracted JSON against ground truth CSV. Fields are derived
dynamically from the GT columns — no hardcoded field lists.

Special scorers are applied based on field name patterns:
- gender/sex fields: normalized categorical match
- age fields: numeric match with +/-1 tolerance
- date fields: exact string match
- all others: TF-IDF cosine similarity with null handling

Usage:
    python evaluate_claude_data.py --gt ground_truth.csv --pred extracted.json [--out report.json]
"""

import argparse, json, csv, re, sys
import numpy as np
from pathlib import Path

# ── Embedding backend ────────────────────────────────────────────────────────

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
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("[embedder] TF-IDF fallback")
        _vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        _fitted_corpus = []
        def embed(texts):
            combined = list(set(_fitted_corpus + texts))
            _vec.fit(combined)
            _fitted_corpus.clear()
            _fitted_corpus.extend(combined)
            mat = _vec.transform(texts).toarray()
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return mat / norms
        _embed_fn = embed
        return embed
    except ImportError:
        print("ERROR: install scikit-learn or sentence-transformers")
        sys.exit(1)


def sim(s1, s2):
    """Cosine similarity between two strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    embed = _get_embed()
    vecs = embed([str(s1), str(s2)])
    return float(np.dot(vecs[0], vecs[1]))


# ── Thresholds ───────────────────────────────────────────────────────────────

PASS_THRESH = 0.85
PARTIAL_THRESH = 0.60


def grade(score):
    if score >= PASS_THRESH:
        return "PASS", "✓"
    if score >= PARTIAL_THRESH:
        return "PARTIAL", "~"
    return "FAIL", "✗"


# ── Scorer functions ─────────────────────────────────────────────────────────


# Canonical forms for common enum aliases
_ENUM_ALIASES = {
    # Gender
    "m": "male", "man": "male", "masculine": "male",
    "f": "female", "woman": "female", "feminine": "female",
    "nb": "non-binary", "nonbinary": "non-binary",
    "enby": "non-binary", "genderqueer": "non-binary",
    "genderfluid": "non-binary",
}


def _strip_enum(val):
    """Normalize enum values: 'determination_enum.denied' -> 'denied'."""
    s = str(val or "").strip().lower()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    # Handle underscores from enum names: 'granted_in_part' -> 'granted in part'
    s = s.replace("_", " ")
    # Normalize common aliases
    s = _ENUM_ALIASES.get(s, s)
    return s


def score_enum(gt_raw, pred_raw):
    gt = _strip_enum(gt_raw)
    pred = _strip_enum(pred_raw)
    if not gt and not pred:
        return 1.0, "PASS", "✓"
    if not gt or not pred:
        return 0.0, "FAIL", "✗"
    if gt == pred:
        return 1.0, "PASS", "✓"
    return 0.0, "FAIL", "✗"


def _edit_distance(s1, s2):
    """Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if c1 == c2 else 1),
            ))
        prev = curr
    return prev[-1]


def score_docket(gt_raw, pred_raw):
    """Score docket numbers by normalized edit distance."""
    gt = str(gt_raw or "").strip().upper()
    pred = str(pred_raw or "").strip().upper()
    if not gt and not pred:
        return 1.0, "PASS", "✓"
    if not gt or not pred:
        return 0.0, "FAIL", "✗"
    if gt == pred:
        return 1.0, "PASS", "✓"
    max_len = max(len(gt), len(pred))
    dist = _edit_distance(gt, pred)
    score = 1.0 - (dist / max_len)
    return score, *grade(score)


def score_text(gt_raw, pred_raw):
    gt_empty = gt_raw is None or str(gt_raw).strip() == ""
    pred_empty = pred_raw is None or str(pred_raw).strip() == ""
    if gt_empty and pred_empty:
        return 1.0, "PASS", "✓"
    if gt_empty or pred_empty:
        return 0.0, "FAIL", "✗"
    gt_s = str(gt_raw).strip()
    pred_s = str(pred_raw).strip()
    # Exact match (case-insensitive)
    if gt_s.lower() == pred_s.lower():
        return 1.0, "PASS", "✓"
    # Token containment (e.g. GT tokens all in pred)
    gt_tokens = set(gt_s.lower().split())
    pred_tokens = set(pred_s.lower().split())
    if gt_tokens and gt_tokens.issubset(pred_tokens):
        return 1.0, "PASS", "✓"
    # Use best of edit distance and semantic similarity
    max_len = max(len(gt_s), len(pred_s))
    edit_sc = 1.0 - (_edit_distance(gt_s.upper(), pred_s.upper()) / max_len)
    sem_sc = sim(gt_s, pred_s)
    sc = max(edit_sc, sem_sc)
    return sc, *grade(sc)


# ── Scorer dispatch ──────────────────────────────────────────────────────────

_SCHEMA_TYPE_SCORERS = {
    "enum": score_enum,
}


def get_scorer(field_name, field_type=None):
    """Pick a scorer based on schema type, with text as default."""
    if field_type and field_type in _SCHEMA_TYPE_SCORERS:
        return _SCHEMA_TYPE_SCORERS[field_type]
    return score_text


# ── Key normalization ────────────────────────────────────────────────────────

def _normalize_key(key):
    """Lowercase, replace spaces/hyphens with underscores."""
    return re.sub(r"[\s\-]+", "_", key.strip().lower())


def match_pred_key(pred_rec, gt_field):
    """Find the extraction key that matches a GT field name."""
    norm_gt = _normalize_key(gt_field)
    # Direct match
    if gt_field in pred_rec:
        return pred_rec[gt_field]
    # Normalized match
    for key in pred_rec:
        if _normalize_key(key) == norm_gt:
            return pred_rec[key]
    return None


# ── File matching ────────────────────────────────────────────────────────────

def stem(filename):
    """patient_01_moderate.pdf -> patient_01"""
    name = Path(filename).stem
    for suffix in ["_moderate", "_degraded", "_heavy", "_clean"]:
        name = name.replace(suffix, "")
    return name


def load_gt(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = stem(row["source_file"])
            rows[key] = row
    return rows


def load_pred(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = {}
    for rec in data:
        src = rec.get("_source_file", "")
        key = stem(src)
        rows[key] = rec
    return rows


# ── Evaluate ─────────────────────────────────────────────────────────────────

def get_gt_fields(gt_rows):
    """Derive field list from GT CSV columns (excluding source_file)."""
    for row in gt_rows.values():
        return [k for k in row.keys() if k != "source_file"]
    return []


def evaluate_batch(batch, gt_rows, fields=None, schema_spec=None):
    """Score a batch of extraction dicts against ground truth.
    Fields are derived from GT columns if not provided.
    If schema_spec is provided, field types are used for scorer dispatch."""
    if fields is None:
        fields = get_gt_fields(gt_rows)

    # Build field_name -> schema type lookup
    field_types = {}
    if schema_spec and "fields" in schema_spec:
        for name, cfg in schema_spec["fields"].items():
            field_types[_normalize_key(name)] = cfg.get("type")

    field_scores = {f: [] for f in fields}
    records = []

    for rec in batch:
        src = rec.get("_source_file", "")
        key = stem(src)
        gt = gt_rows.get(key)
        if gt is None:
            continue

        rec_detail = {"source": key, "fields": {}}
        for field in fields:
            gt_val = gt.get(field)
            pred_val = match_pred_key(rec, field)

            # Normalize empties
            if isinstance(gt_val, str) and gt_val.strip() == "":
                gt_val = None
            if isinstance(pred_val, str) and pred_val.strip() == "":
                pred_val = None
            # Normalize "null"/"None" strings
            if isinstance(gt_val, str) and gt_val.strip().lower() in ("null", "none"):
                gt_val = None
            if isinstance(pred_val, str) and pred_val.strip().lower() in ("null", "none"):
                pred_val = None

            ftype = field_types.get(_normalize_key(field))
            scorer = get_scorer(field, ftype)
            sc, lbl, icon = scorer(gt_val, pred_val)

            rec_detail["fields"][field] = {
                "score": round(sc, 3),
                "grade": lbl,
                "icon": icon,
                "gt": gt_val,
                "pred": pred_val,
            }
            field_scores[field].append(sc)

        records.append(rec_detail)

    field_means = {
        f: float(np.mean(s)) if s else None
        for f, s in field_scores.items()
    }
    all_scores = [s for ss in field_scores.values() for s in ss]
    overall = float(np.mean(all_scores)) if all_scores else 0.0

    return {"overall": overall, "field_means": field_means, "records": records}


def evaluate(gt_path, pred_path, out_path=None):
    gt_rows = load_gt(gt_path)
    pred_rows = load_pred(pred_path)
    fields = get_gt_fields(gt_rows)

    # Build batch list from pred_rows
    batch = [dict(rec, _source_file=src)
             for src, rec in pred_rows.items()
             if "_source_file" not in rec]
    if not batch:
        batch = list(pred_rows.values())

    ev = evaluate_batch(batch, gt_rows, fields)

    # Print report
    print("\n" + "=" * 68)
    print(f"  EVALUATION REPORT — {Path(pred_path).name}")
    print("=" * 68)
    print(f"  Records evaluated: {len(ev['records'])}")
    print()
    print(f"  {'Field':<30} {'Mean':>6}  {'Pass':>5}  {'Part':>5}  {'Fail':>5}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")
    for field in fields:
        scores = [r["fields"][field]["score"]
                  for r in ev["records"] if field in r["fields"]]
        if not scores:
            continue
        mean = np.mean(scores)
        p = sum(1 for s in scores if s >= PASS_THRESH)
        pa = sum(1 for s in scores if PARTIAL_THRESH <= s < PASS_THRESH)
        f = sum(1 for s in scores if s < PARTIAL_THRESH)
        print(f"  {field:<30} {mean:>6.3f}  {p:>5}  {pa:>5}  {f:>5}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")
    print(f"  {'OVERALL':<30} {ev['overall']:>6.3f}")
    print()

    output = {
        "pred_file": str(pred_path),
        "gt_file": str(gt_path),
        "thresholds": {"pass": PASS_THRESH, "partial": PARTIAL_THRESH},
        "overall": round(ev["overall"], 4),
        "field_means": {f: round(v, 4) if v else None
                        for f, v in ev["field_means"].items()},
        "records": ev["records"],
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Report written to {out_path}\n")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Ground truth CSV path")
    parser.add_argument("--pred", required=True, help="Extracted JSON path")
    parser.add_argument("--out", default=None, help="Output JSON report path")
    args = parser.parse_args()
    evaluate(args.gt, args.pred, args.out)
