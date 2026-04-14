"""
benchmark.py — Petey benchmark runner
======================================
Runs extraction benchmarks across models, parsers, and datasets.
No config files needed — everything is specified via CLI.

Schemas and ground truth live in data/ (checked into the repo).
PDFs are downloaded from GCS on first run and cached locally.
Results are saved locally and optionally uploaded to GCS.

Usage:
    # Full sweep
    python benchmark.py --datasets medical,par_simple,par_detailed

    # Specific combo
    python benchmark.py --models gpt-4.1,gpt-5 --parsers datalab --datasets par_simple

    # With redundancy and GCS upload
    python benchmark.py --models gpt-4.1 --datasets medical --runs 3 --gcs

    # Dry run
    python benchmark.py --dry-run

    # Ad hoc single run
    python benchmark.py --models gpt-5.4 --parsers pymupdf --datasets par_simple --runs 1
"""

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_CACHE = ROOT / ".pdf_cache"

GCS_BUCKET = "gs://benchmarks"

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = {
    "medical": {
        "pdf_gcs": f"{GCS_BUCKET}/claude_medical/",
        "pdf_glob": "*.pdf",
        "schema": DATA_DIR / "medical" / "schema.yaml",
        "ground_truth": DATA_DIR / "medical" / "ground_truth.csv",
    },
    "par_simple": {
        "pdf_gcs": f"{GCS_BUCKET}/par_pdf_subset/",
        "pdf_glob": "*.pdf",
        "schema": DATA_DIR / "par_simple" / "schema.yaml",
        "ground_truth": DATA_DIR / "par_simple" / "ground_truth.csv",
    },
    "par_detailed": {
        "pdf_gcs": f"{GCS_BUCKET}/par_pdf_subset/",
        "pdf_glob": "*.pdf",
        "schema": DATA_DIR / "par_detailed" / "schema.yaml",
        "ground_truth": DATA_DIR / "par_detailed" / "ground_truth.csv",
    },
}


def _gcs_client():
    """Get a GCS client (lazy import)."""
    try:
        from google.cloud import storage
        return storage.Client()
    except ImportError:
        return None


def _ensure_pdfs(dataset: str) -> Path:
    """Download PDFs from GCS if not cached locally. Returns local dir."""
    ds = DATASETS[dataset]
    local_dir = PDF_CACHE / dataset
    if local_dir.exists() and any(local_dir.glob(ds["pdf_glob"])):
        return local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    gcs_src = ds["pdf_gcs"]
    print(f"  Downloading PDFs: {gcs_src} → {local_dir}/")

    # Try Python client first, fall back to gcloud CLI
    client = _gcs_client()
    if client:
        bucket_name = GCS_BUCKET.replace("gs://", "").rstrip("/")
        prefix = gcs_src.replace(f"gs://{bucket_name}/", "")
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            fname = os.path.basename(blob.name)
            blob.download_to_filename(str(local_dir / fname))
        n = len(list(local_dir.glob(ds["pdf_glob"])))
        print(f"  Downloaded {n} files")
    else:
        subprocess.run(
            ["gcloud", "storage", "cp", f"{gcs_src}*",
             str(local_dir) + "/"],
            check=True,
        )
        n = len(list(local_dir.glob(ds["pdf_glob"])))
        print(f"  Downloaded {n} files")
    return local_dir


def _upload_to_gcs(local_path: Path, gcs_dest: str):
    """Upload a file to GCS."""
    client = _gcs_client()
    if client:
        bucket_name = GCS_BUCKET.replace("gs://", "").rstrip("/")
        blob_path = gcs_dest.replace(f"gs://{bucket_name}/", "")
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(local_path))
    else:
        subprocess.run(
            ["gcloud", "storage", "cp", str(local_path), gcs_dest],
            check=True,
        )

# ---------------------------------------------------------------------------
# Model and parser lists
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5.4",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "gemini/gemini-2.5-flash",
    "deepseek/deepseek-chat",
]

ALL_PARSERS = [
    "pymupdf",
    "datalab",
    "unstructured",
]

# Friendly names for output paths
MODEL_NAMES = {
    "gemini/gemini-2.5-flash": "gemini-2.5-flash",
    "deepseek/deepseek-chat": "deepseek-chat",
}

PARSER_NAMES = {}


def safe_name(s):
    """Convert model/parser ID to filesystem-safe name."""
    return MODEL_NAMES.get(s, PARSER_NAMES.get(s, s)).replace("/", "-")


# Auth/quota error patterns
_AUTH_PATTERNS = [
    "invalid api key", "invalid x-api-key", "incorrect api key",
    "authentication", "unauthorized", "forbidden",
    "quota", "rate limit", "exceeded", "insufficient",
    "no credits", "billing", "payment required",
    "does not exist or you do not have access",
]


def _is_auth_error(error_str: str) -> bool:
    """Check if an error string indicates an auth or quota problem."""
    lower = error_str.lower()
    return any(p in lower for p in _AUTH_PATTERNS)


def _check_batch_auth(results: list[dict]) -> str | None:
    """If most results failed with auth/quota errors, return the error.

    Returns the error message if >=80% of results have the same auth
    error, None otherwise.
    """
    errors = [r["_error"] for r in results if r.get("_error")]
    if not errors:
        return None
    auth_errors = [e for e in errors if _is_auth_error(e)]
    if len(auth_errors) >= len(results) * 0.8:
        return auth_errors[0]
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


async def extract_dataset(
    pdf_files: list[str],
    schema_model,
    spec: dict,
    model: str,
    parser: str,
    concurrency: int = 10,
    max_retries: int = 2,
) -> tuple[list[dict], dict]:
    """Run extraction and return (results, timing_info).

    Failed files are retried up to max_retries times.

    timing_info contains:
        batch_seconds: total wall clock for the batch
        per_file: {filename: seconds} for each file
    """
    from petey import extract_batch
    from petey.schema import normalize_dates

    per_file_times = {}
    file_start_times = {}

    def on_result(path, data):
        fname = os.path.basename(path)
        elapsed = time.time() - file_start_times.get(path, time.time())
        per_file_times[fname] = round(elapsed, 2)
        status = "x" if data.get("_error") else "."
        print(f"  {status} {fname} ({elapsed:.1f}s)", flush=True)

    # Record start times for each file
    for p in pdf_files:
        file_start_times[p] = time.time()

    t0 = time.time()
    results = await extract_batch(
        pdf_files,
        schema_model,
        model=model,
        parser=parser,
        instructions=spec.get("instructions", ""),
        on_result=on_result,
        concurrency=concurrency,
    )

    # Retry failed files
    for retry in range(1, max_retries + 1):
        failed = [r for r in results if r.get("_error")]
        if not failed:
            break
        # Don't retry auth errors
        auth_err = _check_batch_auth(results)
        if auth_err:
            break

        # Map source_file back to path
        path_by_name = {os.path.basename(p): p for p in pdf_files}
        retry_paths = []
        for r in failed:
            p = path_by_name.get(r.get("_source_file", ""))
            if p:
                retry_paths.append(p)

        if not retry_paths:
            break

        print(f"  ↻ Retry {retry}/{max_retries}: "
              f"{len(retry_paths)} failed files", flush=True)

        for p in retry_paths:
            file_start_times[p] = time.time()

        retry_results = await extract_batch(
            retry_paths,
            schema_model,
            model=model,
            parser=parser,
            instructions=spec.get("instructions", ""),
            on_result=on_result,
            concurrency=concurrency,
        )

        # Replace failed results with retry results
        retry_by_file = {
            r.get("_source_file"): r for r in retry_results
        }
        for i, r in enumerate(results):
            if r.get("_error") and r.get("_source_file") in retry_by_file:
                new = retry_by_file[r["_source_file"]]
                if not new.get("_error"):
                    results[i] = new

    batch_seconds = round(time.time() - t0, 2)

    # Normalize dates
    for rec in results:
        if not rec.get("_error"):
            normalize_dates(rec, spec)

    timing = {
        "batch_seconds": batch_seconds,
        "per_file": per_file_times,
        "file_count": len(pdf_files),
        "files_per_second": round(len(pdf_files) / batch_seconds, 2)
        if batch_seconds > 0
        else 0,
    }

    return results, timing


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    models: list[str],
    parsers: list[str],
    datasets: list[str],
    runs: int = 1,
    output_dir: str = "results",
    concurrency: int = 10,
    dry_run: bool = False,
    limit: int | None = None,
    gcs: bool = False,
    retry_errors: bool = False,
):
    """Run all requested benchmark permutations."""
    from petey.schema import load_schema

    out = Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gcs_results = f"{GCS_BUCKET}/results/"

    # Build permutation list
    combos = []
    for dataset in datasets:
        if dataset not in DATASETS:
            print(f"Unknown dataset: {dataset}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            return
        for parser in parsers:
            for model in models:
                combos.append((dataset, parser, model))

    total_runs = len(combos) * runs
    print(f"Benchmark plan: {len(combos)} combos × {runs} run(s) "
          f"= {total_runs} total extractions")
    print(f"  Models:   {', '.join(models)}")
    print(f"  Parsers:  {', '.join(parsers)}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Output:   {out}/")
    if gcs:
        print(f"  GCS:      {gcs_results}")
    print()

    if dry_run:
        print("DRY RUN — would run:")
        for ds_name, parser, model in combos:
            # Check if PDFs are cached, otherwise just show "?"
            local_dir = PDF_CACHE / ds_name
            ds_info = DATASETS[ds_name]
            if local_dir.exists():
                n_files = len(list(local_dir.glob(ds_info["pdf_glob"])))
            else:
                n_files = "?"
            if limit and isinstance(n_files, int):
                n_files = min(n_files, limit)
            print(f"  {ds_name} / {parser} / {model}  ({n_files} files × {runs} runs)")
        return

    # Ensure PDFs are downloaded for all requested datasets
    pdf_dirs = {}
    for dataset in datasets:
        pdf_dirs[dataset] = _ensure_pdfs(dataset)

    # Run everything
    run_log = {
        "timestamp": timestamp,
        "models": models,
        "parsers": parsers,
        "datasets": datasets,
        "runs": runs,
        "concurrency": concurrency,
        "gcs_upload": gcs,
        "results": [],
    }

    # Track models/parsers with auth failures to skip in future runs
    skip_models = set()   # models with LLM auth/quota errors
    skip_parsers = set()  # parsers with API auth/quota errors

    for run_idx in range(1, runs + 1):
        for combo_idx, (dataset, parser, model) in enumerate(combos, 1):
            # Skip combos with known auth failures
            if model in skip_models:
                print(f"\n  ⊘ Skipping {dataset}/{parser}/{model} "
                      f"(model auth/quota error)")
                continue
            if parser in skip_parsers and parser != "pymupdf":
                print(f"\n  ⊘ Skipping {dataset}/{parser}/{model} "
                      f"(parser auth/quota error)")
                continue

            ds = DATASETS[dataset]
            pdf_dir = pdf_dirs[dataset]
            pdf_files = sorted(str(p) for p in pdf_dir.glob(ds["pdf_glob"]))
            if limit:
                pdf_files = pdf_files[:limit]

            schema_model, spec = load_schema(ds["schema"])

            run_label = (
                f"[run {run_idx}/{runs}] "
                f"[{combo_idx}/{len(combos)}] "
                f"{dataset} / {parser} / {model}"
            )
            print(f"\n{'='*70}")
            print(f"  {run_label}  ({len(pdf_files)} files)")
            print(f"{'='*70}")

            try:
                results, timing = await extract_dataset(
                    pdf_files, schema_model, spec,
                    model=model, parser=parser,
                    concurrency=concurrency,
                )

                # Check for auth/quota errors
                auth_err = _check_batch_auth(results)
                if auth_err:
                    print(f"\n  ⚠ Auth/quota error detected: {auth_err}")
                    # Determine if it's a model or parser issue
                    if parser == "pymupdf":
                        # pymupdf doesn't need auth, so it's the model
                        skip_models.add(model)
                        print(f"  → Skipping model '{model}' "
                              f"for remaining runs")
                    elif "api key" in auth_err.lower() or \
                         "x-api-key" in auth_err.lower():
                        # Parser auth error
                        skip_parsers.add(parser)
                        print(f"  → Skipping parser '{parser}' "
                              f"for remaining runs")
                    else:
                        skip_models.add(model)
                        print(f"  → Skipping model '{model}' "
                              f"for remaining runs")

                # Save results (even if errors, for diagnostics)
                ds_dir = out / dataset
                ds_dir.mkdir(parents=True, exist_ok=True)
                safe_model = safe_name(model)
                safe_parser = safe_name(parser)
                import uuid
                suffix = f"_{uuid.uuid4().hex[:8]}"
                fname = f"{safe_parser}_{safe_model}{suffix}.json"
                out_path = ds_dir / fname

                # Include timing metadata in output
                output = {
                    "meta": {
                        "dataset": dataset,
                        "parser": parser,
                        "model": model,
                        "run": run_idx,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "timing": timing,
                        "auth_error": auth_err,
                    },
                    "results": results,
                }
                with open(out_path, "w") as f:
                    json.dump(output, f, indent=2)

                n_errors = sum(1 for r in results if r.get("_error"))
                print(f"\n  ✓ {len(results)} records, {n_errors} errors, "
                      f"{timing['batch_seconds']}s total")
                print(f"  → {out_path}")

                if gcs:
                    gcs_path = f"{gcs_results}{dataset}/{fname}"
                    _upload_to_gcs(out_path, gcs_path)
                    print(f"  → {gcs_path}")

                run_log["results"].append({
                    "dataset": dataset,
                    "parser": parser,
                    "model": model,
                    "run": run_idx,
                    "file": str(out_path),
                    "records": len(results),
                    "errors": n_errors,
                    "auth_error": auth_err,
                    "timing": timing,
                })

            except Exception as e:
                err_str = str(e)
                print(f"\n  ✗ FAILED: {err_str}")
                if _is_auth_error(err_str):
                    skip_models.add(model)
                    print(f"  → Skipping model '{model}' "
                          f"for remaining runs")
                run_log["results"].append({
                    "dataset": dataset,
                    "parser": parser,
                    "model": model,
                    "run": run_idx,
                    "error": err_str,
                })

    # --- Retry errors from existing results ---
    if retry_errors:
        from petey.schema import load_schema as _ls, normalize_dates

        print(f"\n{'='*70}")
        print("  RETRY: scanning for results with errors")
        print(f"{'='*70}")

        # Download results from GCS if not local
        if gcs:
            for dataset in datasets:
                ds_dir = out / dataset
                ds_dir.mkdir(parents=True, exist_ok=True)
                client = _gcs_client()
                if client:
                    bucket_name = GCS_BUCKET.replace("gs://", "")
                    bucket = client.bucket(bucket_name)
                    prefix = f"results/{dataset}/"
                    for blob in bucket.list_blobs(prefix=prefix):
                        if blob.name.endswith("/"):
                            continue
                        fname = os.path.basename(blob.name)
                        local = ds_dir / fname
                        if not local.exists():
                            blob.download_to_filename(str(local))
                    print(f"  Downloaded results for {dataset}")

        # Scan result files
        retry_combos = []
        for dataset in datasets:
            ds_dir = out / dataset
            if not ds_dir.exists():
                continue
            for result_file in sorted(ds_dir.glob("*.json")):
                if "run_log" in result_file.name:
                    continue
                with open(result_file) as f:
                    data = json.load(f)
                meta = data.get("meta", {})
                results_list = data.get("results", [])
                errors = [r for r in results_list if r.get("_error")]
                if not errors:
                    continue
                # Don't retry auth errors
                if _check_batch_auth(results_list):
                    continue
                retry_combos.append({
                    "file": result_file,
                    "meta": meta,
                    "dataset": meta.get("dataset", dataset),
                    "parser": meta.get("parser", ""),
                    "model": meta.get("model", ""),
                    "run": meta.get("run", 1),
                    "all_results": results_list,
                    "error_files": [
                        r["_source_file"] for r in errors
                        if r.get("_source_file")
                    ],
                })

        if not retry_combos:
            print("  No retryable errors found.")
        else:
            print(f"  Found {len(retry_combos)} result files "
                  f"with errors")
            for rc in retry_combos:
                ds = DATASETS[rc["dataset"]]
                pdf_dir = pdf_dirs.get(rc["dataset"])
                if not pdf_dir:
                    pdf_dir = _ensure_pdfs(rc["dataset"])
                    pdf_dirs[rc["dataset"]] = pdf_dir

                path_by_name = {
                    os.path.basename(str(p)): str(p)
                    for p in pdf_dir.glob(ds["pdf_glob"])
                }
                retry_paths = [
                    path_by_name[f] for f in rc["error_files"]
                    if f in path_by_name
                ]
                if not retry_paths:
                    continue

                schema_model, spec = _ls(ds["schema"])
                print(f"\n  Retrying {len(retry_paths)} files: "
                      f"{rc['dataset']}/{rc['parser']}/{rc['model']} "
                      f"run{rc['run']}")

                try:
                    new_results, _ = await extract_dataset(
                        retry_paths, schema_model, spec,
                        model=rc["model"],
                        parser=rc["parser"],
                        concurrency=concurrency,
                    )
                    # Merge: replace errors with successes
                    new_by_file = {
                        r["_source_file"]: r for r in new_results
                        if not r.get("_error")
                    }
                    fixed = 0
                    for i, r in enumerate(rc["all_results"]):
                        sf = r.get("_source_file", "")
                        if r.get("_error") and sf in new_by_file:
                            rc["all_results"][i] = new_by_file[sf]
                            fixed += 1

                    # Save updated file
                    output = {
                        "meta": rc["meta"],
                        "results": rc["all_results"],
                    }
                    with open(rc["file"], "w") as f:
                        json.dump(output, f, indent=2)

                    remaining = sum(
                        1 for r in rc["all_results"]
                        if r.get("_error")
                    )
                    print(f"  Fixed {fixed}, "
                          f"{remaining} errors remaining")
                    print(f"  → {rc['file']}")

                    if gcs:
                        gcs_path = (
                            f"{gcs_results}{rc['dataset']}/"
                            f"{rc['file'].name}"
                        )
                        _upload_to_gcs(rc["file"], gcs_path)
                        print(f"  → {gcs_path}")

                except Exception as e:
                    print(f"  ✗ Retry failed: {e}")

    if skip_models or skip_parsers:
        print(f"\n⚠ Skipped due to auth/quota errors:")
        if skip_models:
            print(f"  Models:  {', '.join(sorted(skip_models))}")
        if skip_parsers:
            print(f"  Parsers: {', '.join(sorted(skip_parsers))}")

    # Save run log
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"run_log_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"\nRun log: {log_path}")

    if gcs:
        _upload_to_gcs(log_path, f"{gcs_results}{log_path.name}")
        print(f"  → {gcs_results}{log_path.name}")

    # Print timing summary
    print(f"\n{'='*70}")
    print("  TIMING SUMMARY")
    print(f"{'='*70}")
    for entry in run_log["results"]:
        if "error" in entry:
            print(f"  {entry['dataset']}/{entry['parser']}/{entry['model']} "
                  f"run{entry['run']}: FAILED")
        else:
            t = entry["timing"]
            print(f"  {entry['dataset']}/{entry['parser']}/{entry['model']} "
                  f"run{entry['run']}: {t['batch_seconds']}s "
                  f"({t['files_per_second']} files/s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run petey extraction benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sweep with all models and parsers
  python benchmark.py --datasets medical,par_simple

  # Quick test of one combo
  python benchmark.py --models gpt-4.1 --parsers datalab --datasets medical --runs 1

  # Parser comparison on PAR
  python benchmark.py --models gpt-4.1,gpt-5,claude-sonnet-4-6 --parsers pymupdf,datalab,unstructured --datasets par_simple

  # Schema comparison
  python benchmark.py --models gpt-4.1-mini,gpt-4.1,gpt-5,claude-sonnet-4-6 --datasets par_simple,par_detailed --parsers datalab

  # Dry run to see what would execute
  python benchmark.py --dry-run
""",
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        help=f"Model IDs (default: all {len(ALL_MODELS)})",
    )
    parser.add_argument(
        "--parsers", nargs="+", default=ALL_PARSERS,
        help=f"Parser names (default: all {len(ALL_PARSERS)})",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS.keys()),
        help=f"Dataset names (default: all {len(DATASETS)})",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of times to repeat each combo (default: 1)",
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="API concurrency limit (default: 10)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of PDFs per dataset (for quick tests)",
    )
    parser.add_argument(
        "--gcs", action="store_true",
        help="Upload results to GCS bucket after each run",
    )
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="Scan existing results and re-extract failed files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without running",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        models=args.models,
        parsers=args.parsers,
        datasets=args.datasets,
        runs=args.runs,
        output_dir=args.output,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        limit=args.limit,
        gcs=args.gcs,
        retry_errors=args.retry_errors,
    ))


if __name__ == "__main__":
    main()
