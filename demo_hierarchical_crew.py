"."
"""A ONE-row prediction through a Process.hierarchical
crew with an explicit Manager Agent.

Difference from Pattern-2:
- The Manager is a SEPARATE agent above the team (not one of the workers)
- The Manager delegates and validates each worker's result before the next
- All four specialists (User Profiler, Item Analyst, Calibrator, Prediction
  Modeler) are workers with `allow_delegation=False`

Usage:
    uv run python demo_hierarchical_crew.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    r = getattr(_s, "reconfigure", None)
    if r is not None:
        r(encoding="utf-8", errors="replace")

from first_crew.crews import hierarchical_predict_crew


def main() -> None:
    test_file = Path("data/test_review_subset.json")
    row = json.loads(test_file.read_text(encoding="utf-8").splitlines()[0])
    user_id = row["user_id"]
    item_id = row["item_id"]

    print(f"\n[Hierarchical] Running Manager-led Crew for "
          f"user_id={user_id}, item_id={item_id}")
    print(f"[Hierarchical] Ground truth: stars={row.get('stars')}, "
          f"review[:60]={row.get('text', '')[:60]}\n")

    crew = hierarchical_predict_crew()
    result = crew.kickoff(inputs={"user_id": user_id, "item_id": item_id})

    print("\n=== Hierarchical Crew Output ===")
    print(str(result.raw))


if __name__ == "__main__":
    main()
