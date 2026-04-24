"""Review-prediction pipeline as a CrewAI Flow.

Architecture (when CALIBRATOR_ENABLED=1):

    @start -> init_request
                |
         +------+------+
         |             |
    profile_user   profile_item      (run in parallel, both listen to init_request)
         |             |
    calibrate_user     |              (chains after profile_user)
         |             |
         +------+------+
                |
         and_(calibrate_user, profile_item)
                |
            predict        (fires only after both branches are in state)

When CALIBRATOR_ENABLED=0, the calibrate_user node is a no-op pass-through and
the join happens on and_(profile_user, profile_item) as before.

The user and item records are pre-fetched by primary-key lookup in data_store,
then injected directly into each crew's task inputs as JSON strings. RAG is
used by the Item Analyst (review-side) and the Calibrator (review + user-side).
"""

import json
import re

from pydantic import BaseModel

from crewai.flow.flow import Flow, and_, listen, start

from first_crew.crews import (
    _CALIBRATOR_ENABLED,
    calibrator_crew,
    item_profiling_crew,
    review_prediction_crew,
    user_profiling_crew,
)
from first_crew.data_store import (
    get_item,
    get_user,
    get_user_extreme_reviews,
    get_user_reviews,
)


class ReviewPredictionState(BaseModel):
    user_id: str = ""
    item_id: str = ""
    user_profile_report: str = ""
    item_profile_report: str = ""
    calibration_report: str = ""
    user_missing: bool = False
    item_missing: bool = False
    user_avg_stars: float = 3.75
    final_json: str = ""


def _to_text(result) -> str:
    return str(getattr(result, "raw", result))


def _extract_json(text: str) -> dict:
    """Pull the first {...} block out of a noisy LLM response and parse it.

    phi3 often emits preamble, a `## Answer:` header, and markdown fences
    before the JSON object. A greedy regex for the first brace block is
    more reliable than asking the model nicely.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _round_half(x: float) -> float:
    return round(x * 2) / 2


class ReviewPredictionFlow(Flow[ReviewPredictionState]):
    """Orchestrates three mini-crews through a shared state object."""

    @start()
    def init_request(self):
        print(
            f"\n[Flow] Predicting for user_id={self.state.user_id}, "
            f"item_id={self.state.item_id}"
        )

    @listen(init_request)
    def profile_user(self):
        user = get_user(self.state.user_id)
        past = get_user_reviews(self.state.user_id, limit=10)

        self.state.user_missing = not user
        if user:
            self.state.user_avg_stars = float(user.get("average_stars", 3.75))
        else:
            print(f"[Flow] WARNING: user {self.state.user_id} not in subset")
        print(
            f"[Flow] Pre-fetched user ({len(user)} fields) + "
            f"{len(past)} past reviews"
        )

        result = user_profiling_crew().kickoff(
            inputs={
                "user_id": self.state.user_id,
                "user_record_json": json.dumps(user, ensure_ascii=False, indent=2),
                "past_reviews_json": json.dumps(past, ensure_ascii=False, indent=2),
            }
        )
        self.state.user_profile_report = _to_text(result)
        print(
            f"[Flow] User profile ready "
            f"({len(self.state.user_profile_report)} chars)"
        )

    @listen(init_request)
    def profile_item(self):
        item = get_item(self.state.item_id)

        self.state.item_missing = not item
        if not item:
            print(f"[Flow] WARNING: item {self.state.item_id} not in subset")
        print(f"[Flow] Pre-fetched item ({len(item)} fields)")

        result = item_profiling_crew().kickoff(
            inputs={
                "item_id": self.state.item_id,
                "item_record_json": json.dumps(item, ensure_ascii=False, indent=2),
            }
        )
        self.state.item_profile_report = _to_text(result)
        print(
            f"[Flow] Item profile ready "
            f"({len(self.state.item_profile_report)} chars)"
        )

    @listen(profile_user)
    def calibrate_user(self):
        """new agent. Runs after profile_user, in parallel with
        profile_item. If CALIBRATOR_ENABLED=0, or the user is cold-start, it
        writes a sentinel string and returns without an LLM call.
        """
        if not _CALIBRATOR_ENABLED:
            self.state.calibration_report = (
                "**Calibration disabled** — set CALIBRATOR_ENABLED=1 in .env."
            )
            print("[Flow] Calibrator disabled, skipping")
            return
        if self.state.user_missing:
            self.state.calibration_report = (
                "**Calibration unavailable** — user cold-start; downstream "
                "prediction must rely on global priors (EDA knowledge source)."
            )
            print("[Flow] User cold-start, calibration skipped")
            return

        user = get_user(self.state.user_id)
        extremes = get_user_extreme_reviews(
            self.state.user_id, self.state.user_avg_stars, k=3
        )
        print(
            f"[Flow] Fetched {len(extremes['most_negative'])} negative + "
            f"{len(extremes['most_positive'])} positive extreme reviews"
        )
        result = calibrator_crew().kickoff(
            inputs={
                "user_id": self.state.user_id,
                "user_record_json": json.dumps(user, ensure_ascii=False, indent=2),
                "extreme_reviews_json": json.dumps(
                    extremes, ensure_ascii=False, indent=2
                ),
            }
        )
        self.state.calibration_report = _to_text(result)
        print(
            f"[Flow] Calibration report ready "
            f"({len(self.state.calibration_report)} chars)"
        )

    @listen(and_(calibrate_user, profile_item))
    def predict(self):
        result = review_prediction_crew().kickoff(
            inputs={
                "user_profile_report": self.state.user_profile_report,
                "item_profile_report": self.state.item_profile_report,
                "calibration_report": self.state.calibration_report,
            }
        )
        raw = _to_text(result)
        parsed = _extract_json(raw)

        if self.state.user_missing:
            parsed = {
                "stars": 3.75,
                "review": parsed.get("review")
                or "No rating history available; writing a neutral baseline.",
            }
        elif self.state.item_missing:
            parsed["stars"] = _round_half(self.state.user_avg_stars)
            parsed.setdefault(
                "review", "Decent spot, roughly what I expected."
            )

        parsed.setdefault("stars", 3.75)
        parsed.setdefault("review", "")

        self.state.final_json = json.dumps(parsed, ensure_ascii=False)
        print(
            f"[Flow] Prediction ready (stars={parsed['stars']}, "
            f"user_missing={self.state.user_missing}, "
            f"item_missing={self.state.item_missing})"
        )
        return self.state.final_json
