# Yelp Data — Exploratory Data Analysis (EDA)

This document summarises the global statistics of the Yelp dataset that backs this assignment. It is loaded as a `StringKnowledgeSource` at the Crew level so that every agent — the User Profiler, the Item Analyst, and especially the Prediction Modeler — can consult these numbers when deciding how to rate a particular `(user, item)` pair.

**Why this matters.** The Milestone 1 evaluation surfaced a regression-to-mean failure: the model reliably predicted 3.5-4.0 stars even for reviewers who tend toward 1-2 stars or 5 stars. The distribution statistics below are meant to counteract that bias by giving every agent an explicit numerical prior.

## 1. Rating distribution across all reviews

Based on 382,886 reviews from the training set. Note that the distribution is strongly **bimodal-skewed**: 5-star ratings are the plurality, and 1-star ratings are more common than 2-star ratings.

| Stars | Count | Share | Interpretation |
|---|---|---|---|
| 1.0 ★ | 40,979 | 10.7% | Strongly negative — reviewer is complaining about a specific failure |
| 2.0 ★ | 29,909 | 7.8% | Disappointed — reviewer had real expectations that were missed |
| 3.0 ★ | 47,925 | 12.5% | Lukewarm / mixed — reviewer found some good, some bad |
| 4.0 ★ | 100,449 | 26.2% | Positive — reviewer liked it but had minor reservations |
| 5.0 ★ | 163,624 | 42.7% | Enthusiastic — reviewer is actively recommending the place |

**Key takeaway for the Prediction Modeler:** over 45% of real reviews are at an extreme (1-star or 5-star). A prediction that never ventures outside 3.0-4.0 is already wrong on 45% of the rows by construction.

## 2. Review length by star rating

Harsh reviews (1-2 stars) tend to be **longer** than enthusiastic reviews — complainers enumerate every grievance, while fans tend toward punch-lines. Use this to size the generated `review` field appropriately for the predicted rating.

| Stars | Mean words | Median words | 90th-pctl words |
|---|---|---|---|
| 1.0 ★ | 139 | 101 | 288 |
| 2.0 ★ | 144 | 111 | 290 |
| 3.0 ★ | 135 | 106 | 270 |
| 4.0 ★ | 122 | 95 | 245 |
| 5.0 ★ | 98 | 71 | 202 |

## 3. User calibration buckets

Across 26,593 users with a computed `average_stars` field:

| Bucket | Rule | Count | Share |
|---|---|---|---|
| **Strict** | `average_stars` ≤ 3.0 | 4,789 | 18.0% |
| **Balanced** | 3.0 < `average_stars` < 4.0 | 10,092 | 37.9% |
| **Lenient** | `average_stars` ≥ 4.0 | 11,712 | 44.0% |

**Prediction rule-of-thumb.** When the user record provides `average_stars`, bias the prediction *toward* that user's bucket. A Strict reviewer is three times more likely than a Balanced one to give 1-2 stars on any given visit; ignoring that prior is what produces the regression-to-mean failure.

- User review-count range: min 1, median 18, p90 198, max 17,473.
- Users with >500 reviews are rare but important — they are the heavy-tail "power reviewers" whose past history is both richest and most extreme.

## 4. Top business categories

From 22,536 businesses. The top 20 categories account for most of the test set; if the target item's category is in this list, the agent should already have strong priors about typical rating behaviour.

| Rank | Category | Count |
|---|---|---|
| 1 | Restaurants | 6,809 |
| 2 | Food | 3,935 |
| 3 | Shopping | 3,748 |
| 4 | Home Services | 2,732 |
| 5 | Beauty & Spas | 2,344 |
| 6 | Health & Medical | 2,176 |
| 7 | Local Services | 2,004 |
| 8 | Automotive | 1,761 |
| 9 | Nightlife | 1,706 |
| 10 | Bars | 1,531 |
| 11 | Event Planning & Services | 1,325 |
| 12 | Sandwiches | 1,168 |
| 13 | Active Life | 1,099 |
| 14 | Coffee & Tea | 1,051 |
| 15 | American (Traditional) | 957 |
| 16 | Fast Food | 934 |
| 17 | Breakfast & Brunch | 889 |
| 18 | Pizza | 883 |
| 19 | Home & Garden | 865 |
| 20 | Auto Repair | 846 |

- Mean business `stars` across all items: **3.60**. This is the baseline to compare against when the user-record `average_stars` is unavailable (cold-start).

## 5. How to use this knowledge

- **User Profiler agent.** When writing the profile, explicitly classify the user as Strict / Balanced / Lenient using the rule in §3 above, and name the bucket in your Markdown output.
- **Item Analyst agent.** If the business's `stars` is notably above or below the category average in §4, call that out — it's a signal the user may rate differently than category norms suggest.
- **Prediction Modeler agent.** Do not default to 3.75 out of caution. If the User Profile says "Strict, average_stars = 2.1", your prediction should usually be 1.0-3.0, not 3.5-4.0. If the profile says "Lenient, average_stars = 4.5", your prediction should usually be 4.0-5.0. Extreme predictions are frequently correct — §1 shows that 45% of real reviews are at the extremes.
