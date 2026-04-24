"""Batch runner for the Yelp review-prediction Flow.

Reads rows from data/test_review_subset.json, runs ReviewPredictionFlow for each
(user_id, item_id), appends one prediction per row to results/predictions.jsonl,
and computes MAE + text similarity into results/metrics.json.

Resumable: rows already present in predictions.jsonl (matched on user_id+item_id)
are skipped. Use --limit to cap how many of the FIRST N test rows we consider.

Usage:
    uv run first_crew                # default: first 5 rows (smoke test)
    uv run first_crew --limit 50     # Milestone 1 mini-dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Force UTF-8 on the Windows console; CrewAI's event bus prints emoji.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

from first_crew.flow import ReviewPredictionFlow


TEST_FILE = Path("data/test_review_subset.json")
RESULTS_DIR = Path("results")
PREDICTIONS_FILE = RESULTS_DIR / "predictions.jsonl"
METRICS_FILE = RESULTS_DIR / "metrics.json"


def _load_test_rows(limit: int) -> list[dict]:
    rows: list[dict] = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _load_done_keys() -> set[tuple[str, str]]:
    """Return (user_id, item_id) pairs already present in predictions.jsonl."""
    if not PREDICTIONS_FILE.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                keys.add((row["user_id"], row["item_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def _append_prediction(row: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _predict_one(test_row: dict) -> dict:
    """Run the Flow for a single (user_id, item_id) and return a result dict."""
    user_id = test_row["user_id"]
    item_id = test_row["item_id"]
    true_stars = test_row.get("stars")
    true_review = test_row.get("text", "")

    result = {
        "user_id": user_id,
        "item_id": item_id,
        "true_stars": true_stars,
        "true_review": true_review,
        "pred_stars": None,
        "pred_review": "",
        "ok": False,
    }

    try:
        flow = ReviewPredictionFlow()
        flow.kickoff(inputs={"user_id": user_id, "item_id": item_id})
        parsed = json.loads(flow.state.final_json) if flow.state.final_json else {}
        result["pred_stars"] = parsed.get("stars")
        result["pred_review"] = parsed.get("review", "")
        result["ok"] = result["pred_stars"] is not None
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc(limit=3)

    return result


def _compute_metrics() -> dict:
    """Read predictions.jsonl and produce MAE + cosine text similarity."""
    rows: list[dict] = []
    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    ok_rows = [r for r in rows if r.get("ok")]
    n_total = len(rows)
    n_ok = len(ok_rows)

    if not ok_rows:
        return {
            "n_total": n_total,
            "n_ok": 0,
            "success_rate": 0.0,
            "mae_stars": None,
            "text_cosine_mean": None,
        }

    # MAE on stars
    abs_errors = [
        abs(float(r["pred_stars"]) - float(r["true_stars"]))
        for r in ok_rows
        if r.get("true_stars") is not None
    ]
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else None

    # Text similarity — bge-small embeddings, cosine
    text_sim = _text_cosine_mean(
        [r["pred_review"] for r in ok_rows],
        [r["true_review"] for r in ok_rows],
    )

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "success_rate": round(n_ok / n_total, 4),
        "mae_stars": round(mae, 4) if mae is not None else None,
        "text_cosine_mean": round(text_sim, 4) if text_sim is not None else None,
    }


def _text_cosine_mean(preds: list[str], trues: list[str]) -> float | None:
    """Mean cosine similarity between predicted and true review text."""
    pairs = [(p, t) for p, t in zip(preds, trues) if p and t]
    if not pairs:
        return None

    # Lazy import so a single-row smoke run doesn't pay the model-load cost
    # when the user aborts early.
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    pred_texts = [p for p, _ in pairs]
    true_texts = [t for _, t in pairs]
    pred_emb = model.encode(pred_texts, normalize_embeddings=True)
    true_emb = model.encode(true_texts, normalize_embeddings=True)
    sims = np.sum(pred_emb * true_emb, axis=1)
    return float(np.mean(sims))


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Process only the first N rows of the test file (default: 5).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    test_rows = _load_test_rows(args.limit)
    done_keys = _load_done_keys()

    print(f"\nTest rows considered: {len(test_rows)} (limit={args.limit})")
    print(f"Already done: {len(done_keys & {(r['user_id'], r['item_id']) for r in test_rows})}")

    # Pace requests to stay under Nvidia's free-tier rate limit. Set
    # INTER_ROW_SLEEP_SEC=0 to disable; defaults to 10 s, which at ~6 rows/min
    # keeps us comfortably below typical quotas.
    inter_row_sleep = float(os.getenv("INTER_ROW_SLEEP_SEC", "10"))

    for idx, test_row in enumerate(test_rows, start=1):
        key = (test_row["user_id"], test_row["item_id"])
        if key in done_keys:
            print(f"[{idx}/{len(test_rows)}] SKIP (already done) user={key[0]}")
            continue

        print(f"\n[{idx}/{len(test_rows)}] user={key[0]} item={key[1]}")
        result = _predict_one(test_row)
        _append_prediction(result)

        if result["ok"]:
            diff = abs(float(result["pred_stars"]) - float(result["true_stars"]))
            print(
                f"  -> pred={result['pred_stars']} true={result['true_stars']} "
                f"|err|={diff}"
            )
        else:
            print(f"  -> FAILED: {result.get('error', 'unknown')}")

        # Small cooldown between rows — avoids bursting past the API quota.
        # Skip for the final row.
        if inter_row_sleep > 0 and idx < len(test_rows):
            time.sleep(inter_row_sleep)

    print("\nComputing metrics over predictions.jsonl...")
    metrics = _compute_metrics()
    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Metrics ===")
    print(f"  rows_total:        {metrics['n_total']}")
    print(f"  rows_ok:           {metrics['n_ok']}")
    print(f"  success_rate:      {metrics['success_rate']}")
    print(f"  MAE (stars):       {metrics['mae_stars']}")
    print(f"  text cosine mean:  {metrics['text_cosine_mean']}")
    print(f"\nSaved: {PREDICTIONS_FILE}")
    print(f"Saved: {METRICS_FILE}")


if __name__ == "__main__":
    run()
