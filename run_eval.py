"""
run_eval.py — Standalone petey extraction evaluation runner
============================================================
Reads configuration from a JSON file, runs extraction for each config,
evaluates against ground truth, and outputs a summary.

Usage:
    python run_eval.py                      # uses config.json in cwd
    python run_eval.py --config my_cfg.json
    python run_eval.py --config my_cfg.json --dry-run
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from evaluate_claude_data import evaluate_batch, get_gt_fields, load_gt

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── Extraction runner ────────────────────────────────────────────────────────


async def run_config(
    pdf_files: list[str],
    schema_model: type[BaseModel],
    instructions: str,
    parser: str,
    model: str,
    concurrency: int = 10,
) -> tuple[list[dict] | None, float]:
    """Run extraction for a single config. Returns (batch, elapsed_seconds)."""
    from petey import extract_batch

    def on_result(label, result):
        if "_error" in result:
            print(f"  x {label}", flush=True)
        else:
            print(f"  . {label}", flush=True)

    t0 = time.time()
    try:
        batch = await extract_batch(
            pdf_files,
            schema_model,
            model=model,
            parser=parser,
            instructions=instructions,
            on_result=on_result,
            concurrency=concurrency,
        )
        return batch, time.time() - t0
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED after {elapsed:.1f}s: {e}")
        return None, elapsed


async def run_all(cfg: dict, dry_run: bool = False):
    """Main entry point: load config, run extractions, evaluate, report."""
    # Resolve paths
    benchmark_dir = Path(cfg["benchmark_dir"])
    data_dir = benchmark_dir / cfg["data_dir"]
    gt_path = benchmark_dir / cfg["ground_truth"]
    pdf_glob = cfg.get("pdf_glob", "*.pdf")
    pdf_files = sorted(str(p) for p in data_dir.glob(pdf_glob))
    pdf_limit = cfg.get("pdf_limit")
    if pdf_limit:
        pdf_files = pdf_files[:pdf_limit]
    configs = cfg["configs"]
    output_dir = Path(cfg.get("output_dir", "results"))
    save_extractions = cfg.get("save_extractions", True)
    save_report = cfg.get("save_report", True)

    print(f"PDFs: {len(pdf_files)} files from {data_dir}")
    print(f"Ground truth: {gt_path}")
    print(f"Configs: {len(configs)}")
    print()

    if dry_run:
        print("DRY RUN — configs that would be evaluated:")
        for c in configs:
            print(f"  {c['parser']} | {c['model']}")
        return

    # Load schema
    from petey.schema import load_schema

    schema_path = benchmark_dir / cfg["schema"]
    schema_model, spec = load_schema(schema_path)
    print(f"Schema: {schema_path} ({len(spec['fields'])} fields)")

    # Load ground truth
    gt_rows = load_gt(gt_path)
    fields = get_gt_fields(gt_rows)
    print(f"GT fields: {fields}")

    # Run each config
    results = {}
    timings = {}
    run_start = time.time()
    for c in configs:
        parser, model = c["parser"], c["model"]
        label = f"{parser} | {model}"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        batch, elapsed = await run_config(
            pdf_files, schema_model,
            spec.get("instructions", ""),
            parser, model,
            concurrency=c.get("concurrency", 10),
        )
        timings[label] = round(elapsed, 1)
        if batch is not None:
            results[label] = batch
            print(f"  Done — {len(batch)} records in {elapsed:.1f}s")
    total_elapsed = round(time.time() - run_start, 1)

    print(f"\nCompleted {len(results)}/{len(configs)} configs\n")

    # Evaluate
    evals = {}
    for label, batch in results.items():
        ev = evaluate_batch(batch, gt_rows, fields, schema_spec=spec)
        evals[label] = ev
        print(f"{label}  =>  overall: {ev['overall']:.3f}")

    # Build summary
    rows = []
    for label, ev in evals.items():
        parser, model = label.split(" | ")
        row = {
            "parser": parser,
            "model": model,
            "overall": round(ev["overall"], 3),
        }
        for field in fields:
            m = ev["field_means"].get(field)
            row[field] = round(m, 3) if m is not None else None
        all_grades = [
            r["fields"][f]["grade"]
            for r in ev["records"]
            for f in fields
            if f in r["fields"]
        ]
        row["pass"] = sum(1 for g in all_grades if g == "PASS")
        row["partial"] = sum(1 for g in all_grades if g == "PARTIAL")
        row["fail"] = sum(1 for g in all_grades if g == "FAIL")
        rows.append(row)

    if not rows:
        print("\nNo configs completed successfully.")
        return

    summary = pd.DataFrame(rows).sort_values("overall", ascending=False)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(summary.to_string(index=False))

    # Fails detail
    fail_details = []
    for label, ev in evals.items():
        for rec in ev["records"]:
            for field in fields:
                f = rec["fields"].get(field, {})
                if f.get("grade") == "FAIL":
                    fail_details.append({
                        "config": label,
                        "patient": rec["source"],
                        "field": field,
                        "score": f["score"],
                        "gt": f["gt"],
                        "pred": f["pred"],
                    })

    if fail_details:
        fail_df = pd.DataFrame(fail_details)
        print(f"\n{len(fail_df)} total FAILs across all configs")
        print(fail_df.to_string(index=False))

    # Save outputs
    if save_extractions or save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config used for this run
        with open(run_dir / "settings.json", "w") as f:
            json.dump(cfg, f, indent=2)

        if save_extractions:
            for label, batch in results.items():
                safe_label = label.replace(" | ", "_").replace("/", "-")
                out_path = run_dir / f"extraction_{safe_label}.json"
                with open(out_path, "w") as f:
                    json.dump(batch, f, indent=2)
                print(f"  Saved extractions: {out_path}")

        if save_report:
            report = {
                "timestamp": timestamp,
                "pdf_count": len(pdf_files),
                "configs_run": len(results),
                "configs_total": len(configs),
                "total_seconds": total_elapsed,
                "timings": timings,
                "summary": rows,
                "fails": fail_details,
                "evals": {
                    label: {
                        "overall": ev["overall"],
                        "field_means": ev["field_means"],
                    }
                    for label, ev in evals.items()
                },
            }
            report_path = run_dir / "report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"  Saved report: {report_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run petey extraction evaluation"
    )
    parser.add_argument(
        "--config", default="configs/config.json",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print configs without running"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    asyncio.run(run_all(cfg, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
