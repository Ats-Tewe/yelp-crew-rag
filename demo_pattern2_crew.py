"""Lab Task 1 demo — run a ONE-row prediction through a Process.sequential
Pattern-2 (Collaborative Single Task / hub-and-spoke) crew.

Difference from the Flow-based main.py:
- ONE task, not four Flow nodes
- Prediction Modeler owns the task (`allow_delegation=True`)
- User Profiler, Item Analyst, Calibrator sit as peers (delegation off)
- Modeler asks each peer targeted questions via CrewAI's built-in delegation

Usage:
    uv run python demo_pattern2_crew.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    r = getattr(_s, "reconfigure", None)
    if r is not None:
        r(encoding="utf-8", errors="replace")

from first_crew.crews import pattern2_collaborative_crew


def main() -> None:
    # Use the same row 1 of the test file as the Flow-based pipeline does
    test_file = Path("data/test_review_subset.json")
    row = json.loads(test_file.read_text(encoding="utf-8").splitlines()[0])
    user_id = row["user_id"]
    item_id = row["item_id"]

    print(f"\n[Pattern2] Running hub-and-spoke Crew for "
          f"user_id={user_id}, item_id={item_id}")
    print(f"[Pattern2] Ground truth: stars={row.get('stars')}, "
          f"review[:60]={row.get('text', '')[:60]}\n")

    crew = pattern2_collaborative_crew()
    result = crew.kickoff(inputs={"user_id": user_id, "item_id": item_id})

    print("\n=== Pattern-2 Crew Output ===")
    print(str(result.raw))


if __name__ == "__main__":
    main()
