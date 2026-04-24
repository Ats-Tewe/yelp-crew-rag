"."
"""In-memory primary-key indexes over the Yelp subsets.

Replaces the JSONSearchTool RAG pattern. For retrieving a user or item by ID,
a dict lookup is both correct and dramatically more reliable than semantic
search — especially with a small local model that struggles with tool-calling.

Load-on-import is fine for the subset files (~5 MB total). Switching to the
full 323 MB training set later will require a streaming index instead.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# Retrieval pool lives in the full training set. The 5 MB subsets in data/
# cover only ~20-30% of the test keys (81% user-miss, 70% item-miss against
# test_review_subset.json), so we point at the filtered full-set files instead.
_USER_FILE = Path("Complete Training Set/filtered_user.json")
_ITEM_FILE = Path("Complete Training Set/filtered_item.json")
_REVIEW_FILE = Path("Complete Training Set/train_review.json")

# Fields we strip from the user record before sending to the LLM because they
# add tokens without helping the analysis.
_USER_NOISE_FIELDS = frozenset(
    {
        "friends",
        "compliment_more",
        "compliment_profile",
        "compliment_cute",
        "compliment_list",
        "compliment_note",
        "compliment_plain",
        "compliment_cool",
        "compliment_funny",
        "compliment_writer",
        "compliment_photos",
        "compliment_hot",
    }
)

# For past reviews we only need enough to infer style and rating tendencies.
_REVIEW_KEEP_FIELDS = ("stars", "date", "text")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_users = _load_jsonl(_USER_FILE)
_items = _load_jsonl(_ITEM_FILE)
_reviews = _load_jsonl(_REVIEW_FILE)

_user_by_id: dict[str, dict] = {u["user_id"]: u for u in _users}
_item_by_id: dict[str, dict] = {i["item_id"]: i for i in _items}

_reviews_by_user: dict[str, list[dict]] = defaultdict(list)
_reviews_by_item: dict[str, list[dict]] = defaultdict(list)
for _r in _reviews:
    _reviews_by_user[_r["user_id"]].append(_r)
    _reviews_by_item[_r["item_id"]].append(_r)


def get_user(user_id: str) -> dict[str, Any]:
    """Return a cleaned user record (noise fields stripped). Empty dict if missing."""
    raw = _user_by_id.get(user_id)
    if raw is None:
        return {}
    return {k: v for k, v in raw.items() if k not in _USER_NOISE_FIELDS}


def get_item(item_id: str) -> dict[str, Any]:
    """Return the item record. Empty dict if missing."""
    return dict(_item_by_id.get(item_id) or {})


def get_user_reviews(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return the user's most recent reviews (compact fields only)."""
    revs = _reviews_by_user.get(user_id, [])
    revs_sorted = sorted(revs, key=lambda r: r.get("date", ""), reverse=True)
    return [
        {k: r.get(k) for k in _REVIEW_KEEP_FIELDS if k in r}
        for r in revs_sorted[:limit]
    ]


def get_user_extreme_reviews(
    user_id: str, user_avg_stars: float, k: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Return the user's k most NEGATIVE and k most POSITIVE past reviews.

    Used by the Calibrator agent (Lab Task 4) to counteract regression-to-mean.
    "Most extreme" = largest |stars − user_avg_stars|. Returns a dict with
    `most_negative` and `most_positive` lists, each containing up to k review
    records.
    """
    revs = _reviews_by_user.get(user_id, [])
    if not revs:
        return {"most_negative": [], "most_positive": []}

    compact = [
        {k_: r.get(k_) for k_ in _REVIEW_KEEP_FIELDS if k_ in r}
        for r in revs
    ]
    # Split below vs. above the user's average, then sort each half by
    # |deviation| descending.
    below = [r for r in compact if r.get("stars") is not None
             and float(r["stars"]) < user_avg_stars]
    above = [r for r in compact if r.get("stars") is not None
             and float(r["stars"]) > user_avg_stars]
    below.sort(
        key=lambda r: abs(float(r["stars"]) - user_avg_stars), reverse=True
    )
    above.sort(
        key=lambda r: abs(float(r["stars"]) - user_avg_stars), reverse=True
    )
    return {"most_negative": below[:k], "most_positive": above[:k]}


def stats() -> dict[str, int]:
    return {
        "users": len(_user_by_id),
        "items": len(_item_by_id),
        "reviews": len(_reviews),
        "users_with_reviews": len(_reviews_by_user),
        "items_with_reviews": len(_reviews_by_item),
    }
